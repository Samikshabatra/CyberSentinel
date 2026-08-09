"""Input sanitisation and indicator extraction.

Security posture: input is treated as untrusted data, never as code. Nothing in
this module executes, resolves, fetches or renders analyst input - it only
normalises text and extracts indicators with regular expressions.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable

from cybersentinel.utils.config import get_settings

# Control characters other than tab/newline/carriage-return.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_EVENT_SPLIT = re.compile(r"(?im)^\s*(?:event\s*\d+\s*[:.\-]|---+|\d+[.)]\s+)")

IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL_PATTERN = re.compile(r"\b(?:https?|ftp)://[^\s<>\"')]+", re.IGNORECASE)
DEFANGED_URL_PATTERN = re.compile(r"\bhxxps?\[?:\]?//[^\s<>\"')]+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
DOMAIN_PATTERN = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b", re.IGNORECASE
)
HASH_PATTERN = re.compile(r"\b(?:[a-f0-9]{32}|[a-f0-9]{40}|[a-f0-9]{64})\b", re.IGNORECASE)
PORT_PATTERN = re.compile(r"\bport\s+(\d{1,5})\b", re.IGNORECASE)
USER_PATTERN = re.compile(
    r"\b(?:user|username|account|login|for user)\s*[:=]?\s*['\"]?([A-Za-z0-9._\\-]{2,64})['\"]?",
    re.IGNORECASE,
)
HOSTNAME_PATTERN = re.compile(r"\b(?:host|hostname|server|machine)\s*[:=]\s*([A-Za-z0-9._-]{2,64})")

# Words that look like domains but are almost always file names in log text.
_FILE_SUFFIXES = (
    ".exe", ".dll", ".sh", ".py", ".log", ".txt", ".json", ".conf", ".yaml", ".yml", ".zip",
)


class InputValidationError(ValueError):
    """Raised when analyst input cannot be accepted."""


def sanitize_text(text: str, max_chars: int | None = None) -> str:
    """Normalise untrusted text: strip control characters, collapse blank runs, cap length."""
    if text is None:
        raise InputValidationError("input must not be None")

    limit = max_chars if max_chars is not None else get_settings().max_input_chars
    cleaned = _CONTROL_CHARS.sub("", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        raise InputValidationError("input must not be empty")
    if len(cleaned) > limit:
        cleaned = cleaned[:limit]
    return cleaned


def validate_upload_size(size_bytes: int, max_bytes: int | None = None) -> None:
    """Reject oversized uploads before they are read into memory."""
    limit = max_bytes if max_bytes is not None else get_settings().max_upload_bytes
    if size_bytes > limit:
        raise InputValidationError(f"upload exceeds the {limit} byte limit")


def is_public_ipv4(value: str) -> bool:
    """True for a routable public IPv4 address."""
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError:
        return False
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _valid_ips(candidates: Iterable[str]) -> list[str]:
    valid: list[str] = []
    for candidate in candidates:
        try:
            ipaddress.IPv4Address(candidate)
        except ipaddress.AddressValueError:
            continue
        if candidate not in valid:
            valid.append(candidate)
    return valid


def refang(text: str) -> str:
    """Convert defanged indicators (hxxp://, 1[.]2[.]3[.]4) back to plain form.

    Used only to make pattern matching work; nothing is ever fetched.
    """
    result = text.replace("[.]", ".").replace("(.)", ".").replace("[:]", ":")
    return re.sub(r"\bhxxp", "http", result, flags=re.IGNORECASE)


def extract_indicators(text: str) -> dict[str, list[str]]:
    """Extract observable indicators used for correlation and history lookup."""
    refanged = refang(text)

    urls = sorted(set(URL_PATTERN.findall(refanged)))
    emails = sorted(set(EMAIL_PATTERN.findall(refanged)))
    ips = _valid_ips(IPV4_PATTERN.findall(refanged))
    hashes = sorted({match.lower() for match in HASH_PATTERN.findall(refanged)})
    ports = sorted({match for match in PORT_PATTERN.findall(refanged) if int(match) <= 65535})
    users = sorted({match for match in USER_PATTERN.findall(refanged)})
    hosts = sorted(set(HOSTNAME_PATTERN.findall(refanged)))

    # Domains: exclude anything already captured as a URL host, an email domain,
    # an IP address, or an obvious file name.
    url_hosts = {re.sub(r"^\w+://", "", url).split("/")[0].split(":")[0].lower() for url in urls}
    email_domains = {email.split("@")[1].lower() for email in emails}
    domains: list[str] = []
    for candidate in DOMAIN_PATTERN.findall(refanged):
        lowered = candidate.lower()
        if lowered in url_hosts or lowered in email_domains or lowered in {i.lower() for i in ips}:
            continue
        if lowered.endswith(_FILE_SUFFIXES):
            continue
        if lowered not in domains:
            domains.append(lowered)

    return {
        "ips": ips,
        "public_ips": [ip for ip in ips if is_public_ipv4(ip)],
        "urls": urls,
        "domains": sorted(domains),
        "emails": emails,
        "hashes": hashes,
        "ports": ports,
        "users": users,
        "hosts": hosts,
    }


def split_events(text: str) -> list[str]:
    """Split a multi-event submission into individual events.

    Recognises `Event 1:` headers, `---` separators and numbered lists. Falls
    back to one event per non-empty line when the text is clearly a log burst,
    otherwise treats the whole input as a single event.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    # Explicit separators are authoritative and are honoured even for an email.
    parts = [part.strip() for part in _EVENT_SPLIT.split(cleaned) if part and part.strip()]
    if len(parts) > 1:
        return parts

    # An email has a blank line between headers and body by definition, so blank
    # lines must not be read as event boundaries here.
    if looks_like_email(cleaned):
        return [cleaned]

    blocks = [block.strip() for block in re.split(r"\n\s*\n", cleaned) if block.strip()]
    if len(blocks) > 1:
        return blocks

    return [cleaned]


def looks_like_url_only(text: str) -> bool:
    """True when the whole input is a single URL."""
    stripped = refang(text).strip()
    if " " in stripped or "\n" in stripped:
        return False
    return bool(URL_PATTERN.fullmatch(stripped))


def looks_like_log(text: str) -> bool:
    """Heuristic: repeated timestamped or syslog-shaped lines."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    timestamped = sum(
        1
        for line in lines
        if re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", line)
        or re.match(r"^[A-Z][a-z]{2}\s+\d{1,2}\s\d{2}:\d{2}:\d{2}", line)
        or re.search(r"\b(sshd|kernel|nginx|apache|systemd|auditd)\b\[?\d*\]?:", line)
    )
    return timestamped >= max(2, len(lines) // 2)


def looks_like_email(text: str) -> bool:
    """Heuristic: presence of email header fields."""
    header_hits = sum(
        1
        for field in ("from:", "to:", "subject:", "reply-to:", "return-path:")
        if re.search(rf"(?im)^\s*{re.escape(field)}", text)
    )
    return header_hits >= 2
