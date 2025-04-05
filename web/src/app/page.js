'use client'
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useState, useEffect, useCallback } from "react"
import demopng from "../../public/demo.png"

export default function InputFile() {
  const [isUploading, setIsUploading] = useState(false)
  const [preview, setPreview] = useState(demopng)
  const [svgData, setSvgData] = useState([])
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

  const fetchInitialSegment = useCallback(async () => {
    try {
      setIsUploading(true)
      const response = await fetch(demopng.src)
      const blob = await response.blob()
      const file = new File([blob], 'demo.png', { type: 'image/png' })
      await fetchSegement(file)
      
    } catch (error) {
      console.error("分割错误:", error)
    } finally {
      setIsUploading(false)
    }
  }, [])  

  useEffect(() => {
    fetchInitialSegment()
  }, [])  // 移除 fetchInitialSegment 依赖

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0]

    if (!file) return
    await fetchSegement(file)

  }

  const fetchSegement = async (file) => {
    try {
      setIsUploading(true)
      setPreview(URL.createObjectURL(file))

      // 创建表单数据对象
      const formData = new FormData()
      formData.append("file", file)
      console.log(file)

      const response = await fetch("/api/everything", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`上传失败: ${response.statusText}`)
      }

      const result = await response.json()
      console.log("上传成功:", result)
      setSvgData(result.data)  // 保存返回的SVG数据
      // alert("文件上传成功！")
    } catch (error) {
      console.error("上传错误:", error)
      alert("文件上传失败，请重试")
    } finally {
      setIsUploading(false)
    }
  }
  const handleImageLoad = (e) => {
    const img = e.target
    setDimensions({
      width: img.naturalWidth,
      height: img.naturalHeight
    })
  }

  return (
    <div className="flex flex-col gap-6 justify-center items-center w-full h-screen bg-red">
      <div className="flex gap-6 px-6 py-16 mx-96 bg-gray-100 rounded-md">
        <Label htmlFor="picture">Picture</Label>
        <Input
          id="picture"
          type="file"
          accept=".jpg,.jpeg,.png"
          className="w-52 cursor-pointer"
          onChange={handleFileUpload}
          disabled={isUploading}
        />
        {isUploading && <p>segement ing...</p>}
      </div>
      {preview && (
        <div className="relative w-[600px]">
          <img
            id="preview"
            className="w-full"
            src={preview || demopng}
            alt=""
            onLoad={handleImageLoad}  // 添加加载事件处理
          />
          {svgData.length > 0 && dimensions.width > 0 && (
            <svg
              viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
              preserveAspectRatio="xMidYMid meet"
              className="absolute top-0 left-0"
              width="600"
              height={dimensions.height / dimensions.width * 600}
              style={{
                pointerEvents: 'none'
              }}
            >
              {svgData.map((path, index) => (
                <path
                  onClick={() => path.fill = true}
                  className="hover:fill-fuchsia-600 hover:opacity-50"
                  key={index}
                  d={path.svg_path}
                  fill={path.fill?'fill-fuchsia-600':'none'}
                  stroke="rgba(0,0,0,0.1)"
                  strokeWidth="1"
                  vectorEffect="non-scaling-stroke"
                  style={{
                    pointerEvents: 'auto',
                    cursor: 'pointer'
                  }}
                />
              ))}
            </svg>
          )}
        </div>
      )}
    </div>
  )
}