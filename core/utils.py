import os
import asyncio
from urllib.parse import urlparse
import aiohttp
from aiohttp import TCPConnector
from typing import Optional, Dict

async def download_s3_file(
    file_url: str,
    save_path: Optional[str] = None,
    tmp_dir: str = "/tmp",
    chunk_size: int = 1024 * 1024,  # 1MB chunks
    max_retries: int = 3,
    proxy: Optional[str] = None,  # 代理地址，例如 "http://user:pass@proxy_ip:port"
    proxy_headers: Optional[Dict[str, str]] = None,  # 代理认证头
    timeout: int = 300  # 超时时间（秒）
) -> str:
    """
    支持HTTP代理的异步文件下载函数

    :param file_url: 文件URL
    :param save_path: 指定保存路径（可选）
    :param tmp_dir: 默认临时目录
    :param chunk_size: 流式下载分块大小
    :param max_retries: 最大重试次数
    :param proxy: 代理地址，如 "http://user:pass@proxy_ip:port"
    :param proxy_headers: 代理认证头（如需要）
    :param timeout: 请求超时时间（秒）
    :return: 本地文件路径
    """
    parsed_url = urlparse(file_url)
    file_name = os.path.basename(parsed_url.path)
    local_file_path = os.path.join(save_path, file_name) if save_path else os.path.join(tmp_dir, file_name)

    last_error = None
    connector = TCPConnector(limit=2, force_close=True, ssl=False)

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                proxy_auth = None
                if proxy_headers:
                    proxy_auth = aiohttp.BasicAuth(
                        login=proxy_headers.get("Proxy-Authorization", "").split(" ")[-1]
                    ) if "Proxy-Authorization" in proxy_headers else None

                async with session.get(
                    file_url,
                    proxy=proxy,
                    proxy_auth=proxy_auth
                ) as response:
                    if response.status == 200:
                        with open(local_file_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(chunk_size):
                                f.write(chunk)
                        return local_file_path
                    else:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
            continue

    raise last_error if last_error else Exception("All retries failed")
