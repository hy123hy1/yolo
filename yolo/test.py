import hmac
import hashlib
import base64
import time

# AppSecret
app_secret = "Xuod4REFLtWJ4bf2CXfs"

# 请求体
body = '{"pageNo":1,"pageSize":2}'

# 生成 Content-MD5
content_md5 = base64.b64encode(hashlib.md5(body.encode("utf-8")).digest()).decode("utf-8")

# 生成 Date（RFC 格式，示例里是 CST 格式）
date = time.strftime("%a %b %d %H:%M:%S CST %Y", time.localtime())

# 构建待签名字符串（注意不要拼接 query 参数）
method = "POST"
accept = "*/*"
content_type = "application/json"
url_path = "/artemis/api/resource/v1/cameras"

string_to_sign = (
    f"{method}\n"
    f"{accept}\n"
    f"{content_md5}\n"
    f"{content_type}\n"
    f"{date}\n"
    f"{url_path}"
)

# 生成签名
signature = base64.b64encode(
    hmac.new(app_secret.encode("utf-8"),
             string_to_sign.encode("utf-8"),
             hashlib.sha256).digest()
).decode("utf-8")

# 构建 curl 命令
curl_cmd = f"""curl -X POST 'http://10.164.60.12:81{url_path}' \\
  -H 'Accept: */*' \\
  -H 'Content-Type: application/json' \\
  -H 'Content-MD5: {content_md5}' \\
  -H 'Date: {date}' \\
  -H 'X-Ca-Key: 25401611' \\
  -H 'X-Ca-Signature: {signature}' \\
  -d '{body}'"""

print(curl_cmd)
