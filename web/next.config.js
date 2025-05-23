module.exports = {
    async rewrites() {
        return [
            {
                source: '/api/:path*',
                destination: 'http://127.0.0.1:8000/api/:path*',
            },
        ]
    },
    experimental: {
        proxyTimeout: 300_000,
    },

}
