from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from config.settings import settings

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def auth_verify(api_key: str = Security(api_key_header)):
    """
    验证 X-API-KEY 是否与 .env 中配置的 settings.API_KEY 匹配
    """
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API Key")
    return {"api_key": api_key}
