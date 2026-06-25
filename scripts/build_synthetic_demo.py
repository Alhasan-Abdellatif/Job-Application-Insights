"""Build the synthetic demo dataset for the public deployment.

Produces ``data/synthetic/demo.csv`` — ~100 fake application-ack emails
across fictional companies, mixed across the same three templates the
real labelling pipeline knows about (LinkedIn / Workday / generic ATS).
Unlike ``ack_only.csv`` (which is the user's real labelled email and
gitignored), this file is **safe to commit** — every value is invented.

The output schema matches what the JAI parser + structured-table loader
expect: ``From, To, Subject, Date, Body, Company, Role``. Re-running the
script is deterministic (seeded) so the CSV diff is stable.

Run::

    python scripts/build_synthetic_demo.py

The same generation is invoked at Modal deploy time (see ``modal_app.py``)
so the deployed demo always carries this exact dataset.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

# ────────────────────────── inputs ──────────────────────────

OUT_PATH = Path("data/synthetic/demo.csv")
SEED = 20260625  # bump only when intentionally changing the corpus
N_APPLICATIONS = 100

# Fictional companies — names that don't match any real employer the
# repo's reviewers might recognise. Each gets a stable email-domain slug.
COMPANIES: list[tuple[str, str]] = [
    ("Aurora Robotics", "aurora-robotics"),
    ("Quantum Logistics", "quantumlog"),
    ("Pelican Health", "pelicanhealth"),
    ("Northwind Bank", "northwindbank"),
    ("Helix Biosciences", "helixbio"),
    ("Riverpoint Capital", "riverpoint"),
    ("Solstice Energy", "solsticeenergy"),
    ("Magnolia AI", "magnolia-ai"),
    ("Aspen Foundry", "aspenfoundry"),
    ("Citrine Cloud", "citrinecloud"),
    ("Beacon Labs", "beaconlabs"),
    ("Mosaic Insurance", "mosaic-ins"),
    ("Lighthouse Analytics", "lighthouseanalytics"),
    ("Lattice Systems", "latticesys"),
    ("Granite Robotics", "graniterobotics"),
]

ROLES: list[str] = [
    "Software Engineer",
    "Senior Data Scientist",
    "Machine Learning Engineer",
    "Backend Engineer",
    "Platform Engineer",
    "Site Reliability Engineer",
    "Product Designer",
    "Quantitative Researcher",
    "Frontend Engineer",
    "Data Engineer",
    "Research Scientist",
    "ML Research Intern",
]

LOCATIONS: list[str] = [
    "London, United Kingdom",
    "Edinburgh, United Kingdom",
    "Dublin, Ireland",
    "Amsterdam, Netherlands",
    "Berlin, Germany",
    "Zurich, Switzerland",
    "Remote (EMEA)",
    "New York, NY, United States",
    "Boston, MA, United States",
    "San Francisco, CA, United States",
    "Toronto, ON, Canada",
]

TEMPLATES = ("linkedin", "workday", "generic_ats")

DATE_START = datetime(2024, 1, 1, tzinfo=UTC)
DATE_END = datetime(2026, 6, 20, tzinfo=UTC)


@dataclass(frozen=True)
class Row:
    from_: str
    to: str
    subject: str
    date: str
    body: str
    company: str
    role: str
    sort_dt: datetime  # used for chronological ordering, not written to CSV


# ────────────────────────── per-template builders ──────────────────────────


def _random_date(rng: random.Random) -> datetime:
    span_seconds = int((DATE_END - DATE_START).total_seconds())
    return DATE_START + timedelta(seconds=rng.randint(0, span_seconds))


def _rfc2822(dt: datetime) -> str:
    """Emit the same RFC 2822 shape Gmail exports use, so the JAI date
    parser accepts it without special-casing."""
    return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")


def _linkedin(company: str, role: str, dt: datetime, rng: random.Random) -> Row:
    location = rng.choice(LOCATIONS)
    body = (
        f"Hi Alhasan,\n\n"
        f"Your application was sent to {company}.\n\n"
        f"{role}\n"
        f"{company} · {location}\n"
        f"Applied: {dt.strftime('%B %d, %Y')}\n"
        f"View job: https://linkedin.com/jobs/view/{rng.randint(10_000_000, 99_999_999)}\n\n"
        f"The team at {company} will review your application and reach out "
        f"if there's a fit. You can track this on your LinkedIn applications page.\n\n"
        f"— LinkedIn"
    )
    return Row(
        from_="LinkedIn <jobs-noreply@linkedin.com>",
        to="alhasan@example.com",
        subject=f"Alhasan, your application was sent to {company}",
        date=_rfc2822(dt),
        body=body,
        company=company,
        role=role,
        sort_dt=dt,
    )


def _workday(company: str, slug: str, role: str, dt: datetime, rng: random.Random) -> Row:
    requisition = f"R-{rng.randint(10_000, 99_999)}"
    body = (
        f"Dear Alhasan,\n\n"
        f"Thank you for applying to {company}. We have received your application "
        f"for the {role} role ({requisition}) and our talent team will review it.\n\n"
        f"If your background matches what the hiring team is looking for, "
        f"a recruiter will reach out within two to three weeks. Either way, "
        f"you'll receive an update on the outcome.\n\n"
        f"You can check the status of your application at any time by signing in "
        f"to your candidate profile.\n\n"
        f"Best regards,\n"
        f"The {company} Talent Acquisition Team"
    )
    return Row(
        from_=f"{company} <{slug}@myworkday.com>",
        to="alhasan@example.com",
        subject=f"Thank you for Applying, {role}",
        date=_rfc2822(dt),
        body=body,
        company=company,
        role=role,
        sort_dt=dt,
    )


def _generic_ats(company: str, slug: str, role: str, dt: datetime, rng: random.Random) -> Row:
    # Mix two surface forms so the demo isn't monotone.
    if rng.random() < 0.5:
        subject = f"Application Received — {role} at {company}"
    else:
        subject = f"Thanks for applying to {company}!"
    body = (
        f"Hello Alhasan,\n\n"
        f"Thanks for applying to the {role} position at {company}. "
        f"We have received your application and will review your profile shortly.\n\n"
        f"Our typical process: an initial recruiter screen, followed by a technical "
        f"interview with the hiring manager, and a final on-site (or virtual on-site). "
        f"We aim to get back to all candidates within ten working days.\n\n"
        f"In the meantime, you can read more about life at {company} on our careers blog.\n\n"
        f"Warm regards,\n"
        f"The {company} Recruiting Team"
    )
    return Row(
        from_=f"{company} Careers <careers@{slug}.example>",
        to="alhasan@example.com",
        subject=subject,
        date=_rfc2822(dt),
        body=body,
        company=company,
        role=role,
        sort_dt=dt,
    )


# ────────────────────────── build ──────────────────────────


def build_rows(n: int, seed: int) -> list[Row]:
    rng = random.Random(seed)
    rows: list[Row] = []
    for _ in range(n):
        company, slug = rng.choice(COMPANIES)
        role = rng.choice(ROLES)
        dt = _random_date(rng)
        template = rng.choice(TEMPLATES)
        if template == "linkedin":
            rows.append(_linkedin(company, role, dt, rng))
        elif template == "workday":
            rows.append(_workday(company, slug, role, dt, rng))
        else:
            rows.append(_generic_ats(company, slug, role, dt, rng))
    # Chronological order — RFC 2822 strings don't sort lexically,
    # so we use the captured datetime for the sort key.
    rows.sort(key=lambda r: r.sort_dt)
    return rows


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["From", "To", "Subject", "Date", "Body", "Company", "Role"])
        for r in rows:
            writer.writerow([r.from_, r.to, r.subject, r.date, r.body, r.company, r.role])


def main() -> None:
    rows = build_rows(N_APPLICATIONS, SEED)
    write_csv(rows, OUT_PATH)

    print(f"Wrote {len(rows)} synthetic ACK emails to {OUT_PATH}")
    by_company = Counter(r.company for r in rows)
    by_template_via_from = Counter(
        "linkedin"
        if r.from_.startswith("LinkedIn")
        else "workday"
        if "myworkday.com" in r.from_
        else "generic_ats"
        for r in rows
    )
    print(f"  companies: {len(by_company)} unique")
    print(f"  template mix: {dict(by_template_via_from)}")
    print(f"  date range: {rows[0].date}  →  {rows[-1].date}")


if __name__ == "__main__":
    main()
