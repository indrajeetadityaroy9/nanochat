"""
Copyright-header and PII normalization of Software Heritage text (fetch_swh.py): a documented, deterministic equivalent
of RefineCode's unreleased step (arXiv 2411.04905, 2.1.1), not a reproduction of it. In order:
- leading comments (lexed by Pygments for the file's path) that mention a copyright are removed;
- credentials become <SECRET>: PEM private keys (RFC 7468), JSON web tokens (RFC 7519), the password of a URL's user
  information (RFC 3986 characters only, so templates such as {$password} are left), and the values detect-secrets'
  provider detectors find, wherever they occur;
- email addresses become <EMAIL>;
- public IPv4 addresses (global per ipaddress) become <IP>, including dotted version numbers that are one.
Names are not detected. detect-secrets' entropy and keyword detectors are not used: on code they flag nearly every file.
"""

import re
import ipaddress

from pygments.lexers import get_lexer_for_filename
from pygments.token import Comment
from pygments.util import ClassNotFound
from detect_secrets.core.plugins.util import get_mapping_from_secret_type_to_class
from detect_secrets.plugins.base import RegexBasedDetector
from detect_secrets.plugins.basic_auth import BasicAuthDetector
from detect_secrets.plugins.ip_public import IPPublicDetector
from detect_secrets.plugins.jwt import JwtTokenDetector
from detect_secrets.plugins.private_key import PrivateKeyDetector

PRIVATE_KEY = re.compile(r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY[A-Z ]*)-----[\s\S]*?-----END \1-----")
JWT = re.compile(r"eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*")  # header and payload are base64url JSON objects
USERINFO = r"(?:[A-Za-z0-9._~!$&'()*+,;=-]|%[0-9A-Fa-f]{2})"  # RFC 3986: unreserved, sub-delims, percent-encoded
URL_PASSWORD = re.compile(rf"(://{USERINFO}+:)(?:{USERINFO}|:)+@")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
# provider detectors; the formats above cover private keys, web tokens, URL passwords and IP addresses
PROVIDERS = [cls() for cls in get_mapping_from_secret_type_to_class().values() if issubclass(cls, RegexBasedDetector)
             and not issubclass(cls, (BasicAuthDetector, PrivateKeyDetector, JwtTokenDetector, IPPublicDetector))]


def without_copyright_header(text, path):
    if "copyright" not in text.lower():
        return text
    try:
        lexer = get_lexer_for_filename(path)
    except ClassNotFound:
        return text
    end = 0
    for index, token, value in lexer.get_tokens_unprocessed(text):
        if not (token in Comment or value.isspace()):
            break
        end = index + len(value)
    return text[end:] if "copyright" in text[:end].lower() else text


def public_ip(match):
    try:
        return "<IP>" if ipaddress.IPv4Address(match.group()).is_global else match.group()
    except ValueError:  # an octet over 255 or with a leading zero: not an address
        return match.group()


def sanitize(text, path):
    text = without_copyright_header(text, path)
    text = PRIVATE_KEY.sub("<SECRET>", text)
    text = JWT.sub("<SECRET>", text)
    text = URL_PASSWORD.sub(r"\1<SECRET>@", text)
    for secret in {secret for detector in PROVIDERS for secret in detector.analyze_string(text)}:
        text = text.replace(secret, "<SECRET>")
    text = EMAIL.sub("<EMAIL>", text)
    return IPV4.sub(public_ip, text)
