import io
import os
import click
from fastapi.responses import FileResponse
import torch
import numpy as np
import uvicorn
import clip

import export_onnx_model
from fastapi import FastAPI, File, Form
from pydantic import BaseModel
from typing import Sequence, Callable
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
from typing_extensions import Annotated
from threading import Lock
from skimage import measure


class Point(BaseModel):
    x: int
    y: int


class Points(BaseModel):
    points: Sequence[Point]
    points_labels: Sequence[int]


class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class TextPrompt(BaseModel):
    text: str


def segment_image(image_array: np.ndarray, segmentation_mask: np.ndarray):
    segmented_image = np.zeros_like(image_array)
    segmented_image[segmentation_mask] = image_array[segmentation_mask]
    return segmented_image


def retrieve(
    elements: Sequence[np.ndarray],
    search_text: str,
    preprocess: Callable[[Image.Image], torch.Tensor],
    model, device=torch.device('cpu')
) -> torch.Tensor:
    with torch.no_grad():
        preprocessed_images = [preprocess(Image.fromarray(
            image)).to(device) for image in elements]
        tokenized_text = clip.tokenize([search_text]).to(device)
        stacked_images = torch.stack(preprocessed_images)
        image_features = model.encode_image(stacked_images)
        text_features = model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T)
    return probs[:, 0].softmax(dim=-1)


def mask_to_svg(mask: np.ndarray):
    # https://github.com/Kingfish404/segment-anything-webui/issues/14
    # Get mask contours
    contours = measure.find_contours(mask)
    if not contours:
        return ""

    # Generate SVG paths
    paths = []
    for contour in contours:
        # Swap x, y coordinates and round to integers
        contour = np.round(contour[:, [1, 0]]).astype(int)
        # Build the path string, using spaces instead of commas
        path = f"M {contour[0][0]} {contour[0][1]}"
        for point in contour[1:]:
            path += f" L {point[0]} {point[1]}"
        path += " Z"  # Close the path
        paths.append(path)

    return " ".join(paths)

@click.command()
@click.option('--model',
              default='vit_b',
              help='model name',
              type=click.Choice(['vit_b', 'vit_l', 'vit_h']))
@click.option('--model_path', default='models/sam_vit_b_01ec64.pth', help='model path')
@click.option('--port', default=8000, help='port')
@click.option('--host', default='0.0.0.0', help='host')
def main(model, model_path, port, host, ):
    # 强制使用 CPU
    device = torch.device("cpu")
    # 或者如果你有 CUDA GPU，可以尝试使用：
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device)

    build_sam = sam_model_registry[model]
    sam = build_sam(checkpoint=model_path).to(device)
    onnx_model_path = model_path.replace('.pth', '.onnx')
    if not os.path.exists(onnx_model_path):
        try:
            export_onnx_model.export(sam, onnx_model_path)
            print(f"Successfully exported ONNX model to {onnx_model_path}")
        except Exception as e:
            print(f"Warning: Failed to export ONNX model: {e}")
            print("Continuing without ONNX export...")
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
    sam_model_lock = Lock()
    print("fuck")
    print(clip.__file__)  # 这会显示 CLIP 模块的位置
    clip_model, preprocess = clip.load("ViT-B/16", device=device)

    app = FastAPI()

    @app.get('/')
    def index():
        return {"code": 0, "data": "Hello World"}

    @app.get('/sam_vit.onnx')
    def forward_onnx():
        return FileResponse(onnx_model_path)

    def compress_mask(mask: np.ndarray):
        flat_mask = mask.ravel()
        idx = np.flatnonzero(np.diff(flat_mask))
        idx = np.concatenate(([0], idx + 1, [len(flat_mask)]))
        counts = np.diff(idx)
        values = flat_mask[idx[:-1]]
        compressed = ''.join(
            [f"{c}{'T' if v else 'F'}" for c, v in zip(counts, values)])
        return compressed


    def mask_to_svg_path(mask: np.ndarray) -> str:
        """将二维掩码转换为 SVG 路径字符串
        
        使用 Marching Squares 算法提取轮廓，并生成标准的 SVG path 数据
        """
        # 确保掩码是 float32 类型
        mask = mask.astype(np.float32)
        
        # 获取掩码轮廓
        contours = measure.find_contours(mask, 0.5)
        
        if not contours:
            return ""
        
        # 生成 SVG 路径
        paths = []
        for contour in contours:
            # 交换 x,y 坐标
            contour = contour[:, [1, 0]]
            
            if len(contour) < 2:  # 忽略太短的路径
                continue
            
            # 构建标准的 SVG 路径命令
            path = f"M {contour[0][0]},{contour[0][1]}"  # 移动到起始点
            
            # 添加线段命令
            for point in contour[1:]:
                path += f" L {point[0]},{point[1]}"
            
            path += " Z"  # 闭合路径
            paths.append(path)
        
        return " ".join(paths)

    @app.post('/api/point')
    async def api_points(
            file: Annotated[bytes, File()],
            points: Annotated[str, Form(...)],
    ):
        ps = Points.parse_raw(points)
        input_points = np.array([[p.x, p.y] for p in ps.points])
        input_labels = np.array(ps.points_labels)
        # 在内存中处理图片
        image_data = Image.open(io.BytesIO(file))
        image_array = np.array(image_data)
        with sam_model_lock:
            predictor.set_image(image_array)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            predictor.reset_image()
        masks = [
            {
                "segmentation": compress_mask(np.array(mask)),
                # 添加 SVG 路径
                "svg_path": mask_to_svg_path(mask),
                "stability_score": float(scores[idx]),
                "bbox": [0, 0, 0, 0],
                "area": np.sum(mask).item(),
            }
            for idx, mask in enumerate(masks)
        ]
        masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)
        return {"code": 0, "data": masks[:]}

    @app.post('/api/box')
    async def api_box(
        file: Annotated[bytes, File()],
        box: Annotated[str, Form(...)],
    ):
        b = Box.parse_raw(box)
        input_box = np.array([b.x1, b.y1, b.x2, b.y2])
        image_data = Image.open(io.BytesIO(file))
        image_array = np.array(image_data)
        with sam_model_lock:
            predictor.set_image(image_array)
            masks, scores, logits = predictor.predict(
                box=input_box,
                multimask_output=False,
            )
            predictor.reset_image()
        masks = [
            {
                "segmentation": compress_mask(np.array(mask)),
                "stability_score": float(scores[idx]),
                "bbox": [0, 0, 0, 0],
                "area": np.sum(mask).item(),
            }
            for idx, mask in enumerate(masks)
        ]
        masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)
        return {"code": 0, "data": masks[:]}

    @app.post('/api/everything')
    async def api_everything(file: Annotated[bytes, File()]):
        print("开始处理图像分割请求")
        image_data = Image.open(io.BytesIO(file))
        image_array = np.array(image_data)
        print(f"图像尺寸: {image_array.shape}")
        
        print("开始生成分割掩码...")
        masks = mask_generator.generate(image_array)
        print(f"生成掩码数量: {len(masks)}")
        
        print("对掩码按稳定性评分排序...")
        arg_idx = np.argsort([mask['stability_score']
                              for mask in masks])[::-1].tolist()
        masks = [masks[i] for i in arg_idx]
        
        print("压缩掩码数据并生成SVG路径...")
        for i, mask in enumerate(masks):
            # 只保留 bbox 和 svg_path
            simplified_mask = {
                'bbox': mask['bbox'],
                'svg_path': mask_to_svg_path(mask['segmentation'])
            }
            masks[i] = simplified_mask
        
        print(f"处理完成，返回 {len(masks)} 个掩码结果")
        return {"code": 0, "data": masks[:]}

    @app.post('/api/clip')
    async def api_clip(
            file: Annotated[bytes, File()],
            prompt: Annotated[str, Form(...)],
    ):
        text_prompt = TextPrompt.parse_raw(prompt)
        image_data = Image.open(io.BytesIO(file))
        image_array = np.array(image_data)
        masks = mask_generator.generate(image_array)
        cropped_boxes = []
        for mask in masks:
            bobx = [int(x) for x in mask['bbox']]
            cropped_boxes.append(segment_image(image_array, mask["segmentation"])[
                bobx[1]:bobx[1] + bobx[3], bobx[0]:bobx[0] + bobx[2]])
        scores = retrieve(cropped_boxes, text_prompt.text,
                          model=clip_model, preprocess=preprocess, device=device)
        top = scores.topk(5)
        masks = [masks[i] for i in top.indices]
        for mask in masks:
            mask['segmentation'] = compress_mask(mask['segmentation'])

            
        return {"code": 0, "data": masks[:]}

    @app.post('/api/embedding')
    async def api_embedding(
        file: Annotated[bytes, File()],
    ):
        image_data = Image.open(io.BytesIO(file))
        image_array = np.array(image_data)
        with sam_model_lock:
            predictor.set_image(image_array)
            image_embedding = predictor.get_image_embedding()
            predictor.reset_image()
        print(image_embedding.shape)
        return {"code": 0, "data": image_embedding.tolist()}

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()