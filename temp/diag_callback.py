"""
MVP -> DataHub callback connectivity diagnostic.
"""

import sys
import socket
import subprocess
from urllib.parse import urlparse

TARGET = sys.argv[1] if len(sys.argv) > 1 else None


def find_stored_callback_url():
    import json
    from pathlib import Path
    fixtures_dir = Path(__file__).resolve().parent.parent / "test" / "fixtures"
    candidates = sorted(fixtures_dir.glob("request_*_20tasks.json"))
    if not candidates:
        candidates = sorted(fixtures_dir.glob("request_*.json"))
    if not candidates:
        return None
    latest = candidates[-1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        return data.get("callback_url")
    except Exception:
        return None


def check_dns(hostname):
    print(f"\n[1] DNS resolve: {hostname}")
    try:
        addrs = socket.getaddrinfo(hostname, None)
        ips = list(set(a[4][0] for a in addrs))
        print(f"    [OK] resolved -> {', '.join(ips)}")
        return ips
    except socket.gaierror as e:
        print(f"    [FAIL] resolve failed: {e}")
        print(f"    [SUGGEST] add '{hostname}' to hosts or Docker network")
        return []


def check_tcp(hostname, port):
    print(f"\n[2] TCP connect: {hostname}:{port}")
    try:
        sock = socket.create_connection((hostname, port), timeout=5)
        sock.close()
        print(f"    [OK] TCP connected")
        return True
    except socket.timeout:
        print(f"    [FAIL] timeout (5s) -- firewall or routing issue")
        return False
    except ConnectionRefusedError:
        print(f"    [FAIL] connection refused -- DataHub not running or wrong port")
        return False
    except socket.gaierror:
        print(f"    [FAIL] hostname not resolved -- DNS issue")
        return False
    except OSError as e:
        print(f"    [FAIL] connection error: {e}")
        return False


def check_http(url):
    print(f"\n[3] HTTP POST: {url}")
    import http.client
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"

    try:
        conn = http.client.HTTPConnection(hostname, port, timeout=10)
        body = '{"run_id":"diag","task_id":"diag","status":"completed","frames":[]}'
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")[:200]
        if 200 <= resp.status < 300:
            print(f"    [OK] callback success: HTTP {resp.status} {body}")
        else:
            print(f"    [WARN] callback returned: HTTP {resp.status} {body}")
        conn.close()
        return True
    except Exception as e:
        print(f"    [FAIL] HTTP request failed: {e}")
        return False


def show_network_info():
    print(f"\n[4] Local network info")
    hostname = socket.gethostname()
    print(f"    hostname: {hostname}")
    try:
        local_ip = socket.gethostbyname(hostname)
        print(f"    local IP: {local_ip}")
    except Exception:
        print(f"    local IP: (unknown)")

    try:
        import platform
        if platform.system() == "Windows":
            result = subprocess.run(
                ["ipconfig"], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("IPv4"):
                    print(f"    {stripped}")
        else:
            result = subprocess.run(
                ["ip", "addr", "show"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "inet " in line:
                    print(f"    {line.strip()}")
    except Exception:
        print(f"    (cannot enumerate interfaces)")


def suggest_fixes(hostname, dns_ok, tcp_ok, http_ok):
    print(f"\n[5] Conclusion")
    if http_ok:
        print("    [OK] DataHub callback URL is reachable, no fix needed")
        return

    if not dns_ok:
        print(f"    [FAIL] hostname '{hostname}' cannot be resolved")
        print(f"    [FIX 1] Add hosts entry:")
        print(f"        Windows: edit C:\\Windows\\System32\\drivers\\etc\\hosts")
        print(f"        Linux:   edit /etc/hosts")
        print(f"        Add: <DataHub IP>  {hostname}")
        print(f"    [FIX 2] Use DataHub's actual IP as callback_url:")
        print(f"        DataHub sets DATAHUB_SELF_URL=http://<actual-IP>:8000")
    elif not tcp_ok:
        print(f"    [FAIL] {hostname}:8000 TCP unreachable")
        print(f"    [CHECK]")
        print(f"        - Is DataHub service running?")
        print(f"        - Is firewall allowing port 8000?")
        print(f"        - Are containers on the same network?")
    elif not http_ok:
        print(f"    [FAIL] HTTP POST failed")
        print(f"    [CHECK] callback path and DataHub service status")


def main():
    print("=" * 60)
    print("MVP -> DataHub Callback Connectivity Diagnostic")
    print("=" * 60)

    url = TARGET
    if not url:
        url = find_stored_callback_url()
        if url:
            print(f"\n[fixture] callback_url: {url}")
        else:
            url = "http://datahub:8000/api/annotation_callback"
            print(f"\n[default] callback_url: {url}")

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    port = parsed.port or 80

    print(f"\nTarget: {url}")
    print(f"Host:   {hostname}:{port}")

    dns_ok = bool(check_dns(hostname))
    tcp_ok = False
    http_ok = False

    if dns_ok:
        tcp_ok = check_tcp(hostname, port)
        if tcp_ok:
            http_ok = check_http(url)

    show_network_info()
    suggest_fixes(hostname, dns_ok, tcp_ok, http_ok)

    print()
    return 0 if http_ok else 1


if __name__ == "__main__":
    sys.exit(main())
