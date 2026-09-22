"""
ClaimGraph AI — Synthetic Data Generator
=========================================
Ye script poora synthetic dataset generate karta hai — clean population pehle,
phir fraud rings inject karta hai, phir legitimate noise add karta hai.

Usage:
    python scripts/generate_data.py --customers 8000 --policies 15000 --claims 25000 --seed 42

Interview story:
    "I generated the clean population first with independent relationships,
    then deliberately injected 20 fraud rings by reusing entities (shared devices,
    shared repair shops, shared payment accounts, coordinated timing). I also
    added legitimate noise — real families share addresses, popular repair shops
    legitimately serve many customers — so the system has to reason about
    degree and pattern, not just 'any shared entity = fraud'."

Output files (data/generated/):
    - customers.csv, policies.csv, claims.csv
    - devices.csv, addresses.csv, repair_shops.csv, payment_accounts.csv
    - fraud_ring_ground_truth.csv (NEVER joined to features)
    - data_generation_report.json (metadata about the run)
"""

import argparse
import json
import os
import random
import string
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Project root add karo path mein
sys.path.insert(0, str(Path(__file__).parent.parent))


# === Helper Functions ===

def generate_id(prefix: str, number: int) -> str:
    """ID generate karo — format: PREFIX-XXXXXX (6 digits, zero-padded)"""
    return f"{prefix}-{number:06d}"


def random_date(start: date, end: date, rng: np.random.Generator) -> date:
    """Random date generate karo given range mein"""
    delta = (end - start).days
    random_days = int(rng.integers(0, max(delta, 1)))
    return start + timedelta(days=random_days)


def random_phone(rng: np.random.Generator) -> str:
    """Random US phone number"""
    return f"{rng.integers(200,999)}{rng.integers(200,999)}{rng.integers(1000,9999)}"


def random_email(first: str, last: str, rng: np.random.Generator) -> str:
    """Realistic email generate karo"""
    domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"]
    separator = rng.choice([".", "_", ""])
    suffix = rng.integers(1, 999)
    domain = rng.choice(domains)
    return f"{first.lower()}{separator}{last.lower()}{suffix}@{domain}"


# === Data Lists (realistic names, brands, etc.) ===

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Raj", "Priya", "Amit", "Neha", "Vikram", "Anjali", "Rahul", "Pooja",
    "Wei", "Mei", "Carlos", "Maria", "Ahmed", "Fatima", "Yuki", "Hiroshi"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Patel", "Kumar", "Singh", "Shah", "Chen", "Wang", "Kim", "Park", "Tanaka"
]

STREET_NAMES = [
    "Main St", "Oak Ave", "Elm St", "Park Blvd", "Cedar Ln", "Maple Dr",
    "Pine St", "Washington Ave", "Lake Rd", "Hill St", "River Rd", "Forest Dr",
    "Sunset Blvd", "Broadway", "Market St", "Spring St", "Highland Ave"
]

CITIES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"), ("Houston", "TX"),
    ("Phoenix", "AZ"), ("Philadelphia", "PA"), ("San Antonio", "TX"), ("San Diego", "CA"),
    ("Dallas", "TX"), ("San Jose", "CA"), ("Austin", "TX"), ("Jacksonville", "FL"),
    ("Fort Worth", "TX"), ("Columbus", "OH"), ("Charlotte", "NC"), ("Indianapolis", "IN"),
    ("San Francisco", "CA"), ("Seattle", "WA"), ("Denver", "CO"), ("Nashville", "TN"),
    ("Miami", "FL"), ("Atlanta", "GA"), ("Portland", "OR"), ("Boston", "MA")
]

DEVICE_BRANDS = {
    "smartphone": [("Apple", "iPhone 14"), ("Apple", "iPhone 15"), ("Samsung", "Galaxy S23"),
                   ("Samsung", "Galaxy S24"), ("Google", "Pixel 8"), ("OnePlus", "12")],
    "laptop": [("Apple", "MacBook Air M2"), ("Dell", "XPS 15"), ("HP", "Spectre x360"),
               ("Lenovo", "ThinkPad X1"), ("ASUS", "ZenBook 14")],
    "tablet": [("Apple", "iPad Pro"), ("Samsung", "Galaxy Tab S9"), ("Microsoft", "Surface Pro 9")],
    "smartwatch": [("Apple", "Watch Series 9"), ("Samsung", "Galaxy Watch 6"), ("Garmin", "Venu 3")],
    "television": [("Samsung", "QLED 65"), ("LG", "OLED C3"), ("Sony", "Bravia XR")],
    "appliance": [("LG", "Washer WM4500"), ("Samsung", "Refrigerator RF28"), ("Dyson", "V15 Detect")]
}

REPAIR_SHOP_NAMES = [
    "QuickFix Electronics", "TechRepair Pro", "Device Doctor", "FixIt Hub",
    "SmartRepair Center", "Gadget Guru", "iRepair Station", "TechSolutions",
    "Mobile Medics", "Screen Savers", "ByteFix", "CircuitMasters",
    "DeviceCare Plus", "RepairZone", "TechRevive", "GadgetFixers",
    "ProTech Repairs", "Digital Surgeons", "FixMyDevice", "ElectroFix"
]

CLAIM_DESCRIPTIONS = {
    "accidental_damage": [
        "Device dropped on concrete floor, screen completely shattered",
        "Accidentally knocked device off table, dent on corner and screen crack",
        "Device fell from pocket while running, multiple scratches and broken glass",
        "Child dropped the device on tile floor, touchscreen unresponsive",
        "Device slipped from hand in parking lot, back panel cracked"
    ],
    "mechanical_failure": [
        "Device stopped turning on suddenly, no response to charging",
        "Power button stuck, cannot turn device on or off",
        "Speakers producing distorted sound, audio hardware failure",
        "Camera module not functioning, shows black screen",
        "Hinge mechanism broken, screen wobbly and disconnecting"
    ],
    "theft": [
        "Device stolen from car, filed police report #PR-{num}",
        "Bag with device snatched at train station, police report filed",
        "Device pickpocketed in crowded area, report #PR-{num}",
        "Home burglary, device among stolen items, police report #PR-{num}",
        "Device stolen from gym locker, filed report with facility"
    ],
    "water_damage": [
        "Device fell into swimming pool, not turning on",
        "Coffee spill on device, keyboard and trackpad not working",
        "Device exposed to rain, screen flickering and touch issues",
        "Device dropped in sink, charging port corroded",
        "Water leaked into bag during storm, device screen has water marks"
    ],
    "screen_damage": [
        "Screen cracked from edge to edge after minor drop",
        "LCD bleeding visible on left side of screen",
        "Touch screen unresponsive in bottom half after impact",
        "Dead pixels spreading across display, started as small cluster",
        "Screen separated from body at the top edge"
    ],
    "battery_issue": [
        "Battery drains from 100% to 0% in 2 hours, device is 6 months old",
        "Device overheating during normal use, battery swelling visible",
        "Battery not holding charge, shuts down at 40% randomly",
        "Charging takes over 8 hours for full charge, battery degraded",
        "Device randomly shuts down, battery health shows 52%"
    ],
    "software_issue": [
        "Device stuck in boot loop, factory reset not working",
        "Operating system corrupted after update, data inaccessible",
        "Device freezing every 10 minutes, requires hard restart",
        "Apps crashing continuously, device unusable for work",
        "Firmware update bricked the device, won't boot past logo"
    ]
}

BANK_NAMES = ["Chase", "Bank of America", "Wells Fargo", "Citibank", "US Bank",
              "PNC Bank", "Capital One", "TD Bank", "Ally Bank", "Discover"]


class SyntheticDataGenerator:
    """
    Main generator class — clean data pehle, fraud rings baad mein.

    Design philosophy (interview ke liye):
    1. Clean population independently generate hoti hai — koi shared entities nahi
       (except families sharing addresses — wo legitimate hai)
    2. Fraud rings SEPARATELY inject hote hain — deliberately entity reuse karke
    3. Legitimate noise bhi add hota hai — popular repair shops, family addresses
    4. Ground truth ALAG file mein — features se physically separated
    """

    def __init__(self, n_customers: int, n_policies: int, n_claims: int, seed: int):
        self.n_customers = n_customers
        self.n_policies = n_policies
        self.n_claims = n_claims
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Entity counters — ID generation ke liye
        self.address_counter = 0
        self.device_counter = 0
        self.customer_counter = 0
        self.policy_counter = 0
        self.claim_counter = 0
        self.repair_shop_counter = 0
        self.payment_counter = 0
        self.ring_counter = 0

        # Storage
        self.addresses = []
        self.devices = []
        self.customers = []
        self.policies = []
        self.claims = []
        self.repair_shops = []
        self.payment_accounts = []
        self.fraud_ground_truth = []

        # Fraud ring tracking
        self.fraud_claim_ids = set()  # Ye claims fraud rings mein hain

    def _next_id(self, prefix: str) -> str:
        """Auto-increment ID generator"""
        counter_name = {
            "ADR": "address", "DEV": "device", "CUS": "customer",
            "POL": "policy", "CLM": "claim", "REP": "repair_shop",
            "PAY": "payment", "RING": "ring"
        }
        attr = f"{counter_name[prefix]}_counter"
        current = getattr(self, attr) + 1
        setattr(self, attr, current)
        if prefix == "RING":
            return f"{prefix}-{current:03d}"
        return f"{prefix}-{current:06d}"

    # =============================================================
    # STEP 1: Generate Clean Population
    # =============================================================

    def generate_addresses(self) -> None:
        """
        Addresses generate karo — roughly 1 per 2 customers.
        Kuch addresses shared honge (family members) — ye LEGITIMATE hai.
        """
        n_addresses = self.n_customers // 2 + self.n_customers // 10
        print(f"📍 Generating {n_addresses} addresses...")

        for _ in range(n_addresses):
            city, state = self.rng.choice(CITIES)
            self.addresses.append({
                "address_id": self._next_id("ADR"),
                "street": f"{self.rng.integers(100, 9999)} {self.rng.choice(STREET_NAMES)}",
                "city": city,
                "state": state,
                "zip_code": f"{self.rng.integers(10000, 99999)}",
                "country": "US"
            })

    def generate_payment_accounts(self) -> None:
        """Payment accounts — 1 per customer initially, fraud rings baad mein share karenge"""
        n_accounts = self.n_customers
        print(f"💳 Generating {n_accounts} payment accounts...")

        for _ in range(n_accounts):
            self.payment_accounts.append({
                "payment_account_id": self._next_id("PAY"),
                "account_type": self.rng.choice(["bank_account", "credit_card", "digital_wallet"]),
                "account_last_four": f"{self.rng.integers(1000, 9999)}",
                "bank_name": self.rng.choice(BANK_NAMES)
            })

    def generate_repair_shops(self) -> None:
        """
        Repair shops — proportional to claims (1 per ~10 claims).
        Kuch shops popular hain legitimately — high volume ≠ always fraud.
        """
        n_shops = max(self.n_claims // 10, 20)
        print(f"🔧 Generating {n_shops} repair shops...")

        for i in range(n_shops):
            shop_address_id = self.rng.choice(self.addresses)["address_id"]
            self.repair_shops.append({
                "repair_shop_id": self._next_id("REP"),
                "name": self.rng.choice(REPAIR_SHOP_NAMES) + f" #{i+1}",
                "address_id": shop_address_id,
                "specialty": self.rng.choice(["electronics", "appliances", "general"]),
                "rating": round(float(self.rng.uniform(2.5, 5.0)), 1),
                "is_authorized": bool(self.rng.random() > 0.2)  # 80% authorized
            })

    def generate_devices(self) -> None:
        """Devices — 1 per customer initially, unique serial numbers"""
        n_devices = self.n_customers
        print(f"📱 Generating {n_devices} devices...")

        for _ in range(n_devices):
            device_type = self.rng.choice(list(DEVICE_BRANDS.keys()))
            brand, model = self.rng.choice(DEVICE_BRANDS[device_type])
            serial = ''.join(self.rng.choice(list(string.ascii_uppercase + string.digits))
                           for _ in range(12))
            purchase_date = random_date(date(2021, 1, 1), date(2024, 6, 1), self.rng)

            # Price range device type ke hisaab se
            price_ranges = {
                "smartphone": (300, 1500), "laptop": (500, 3000),
                "tablet": (200, 1500), "smartwatch": (150, 800),
                "television": (400, 3000), "appliance": (200, 2000)
            }
            min_p, max_p = price_ranges[device_type]

            self.devices.append({
                "device_id": self._next_id("DEV"),
                "device_type": device_type,
                "brand": brand,
                "model": model,
                "serial_number": serial,
                "purchase_date": str(purchase_date),
                "purchase_price": round(float(self.rng.uniform(min_p, max_p)), 2)
            })

    def generate_customers(self) -> None:
        """
        Customers generate karo — har ek ko address aur payment account assign karo.
        Family members DELIBERATELY same address share karte hain (legitimate noise).
        """
        print(f"👤 Generating {self.n_customers} customers...")

        address_ids = [a["address_id"] for a in self.addresses]
        payment_ids = [p["payment_account_id"] for p in self.payment_accounts]

        for i in range(self.n_customers):
            first = self.rng.choice(FIRST_NAMES)
            last = self.rng.choice(LAST_NAMES)

            # ~15% customers family members hain — same address share karte hain
            # Ye LEGITIMATE hai, fraud nahi — system ko ye distinguish karna hai
            if i > 0 and self.rng.random() < 0.15:
                # Pichle kisi customer ka address reuse karo (family)
                family_idx = int(self.rng.integers(0, len(self.customers)))
                addr_id = self.customers[family_idx]["address_id"]
            else:
                addr_id = self.rng.choice(address_ids)

            self.customers.append({
                "customer_id": self._next_id("CUS"),
                "first_name": first,
                "last_name": last,
                "email": random_email(first, last, self.rng),
                "phone": random_phone(self.rng),
                "date_of_birth": str(random_date(date(1960, 1, 1), date(2004, 1, 1), self.rng)),
                "address_id": addr_id,
                "payment_account_id": payment_ids[i] if i < len(payment_ids) else self.rng.choice(payment_ids),
                "created_at": str(datetime.now())
            })

    def generate_policies(self) -> None:
        """
        Policies — multiple policies per customer allowed.
        Har policy ek device se linked hai.
        """
        print(f"📋 Generating {self.n_policies} policies...")

        customer_ids = [c["customer_id"] for c in self.customers]
        device_ids = [d["device_id"] for d in self.devices]

        for _ in range(self.n_policies):
            cust_id = self.rng.choice(customer_ids)
            dev_id = self.rng.choice(device_ids)
            start = random_date(date(2022, 1, 1), date(2024, 6, 1), self.rng)
            duration_days = int(self.rng.choice([365, 730, 1095]))  # 1, 2, or 3 years
            end = start + timedelta(days=duration_days)
            policy_type = self.rng.choice(["basic", "standard", "premium", "extended"])

            premium_ranges = {"basic": (50, 150), "standard": (100, 300),
                            "premium": (200, 500), "extended": (300, 700)}
            coverage_ranges = {"basic": (500, 1500), "standard": (1000, 3000),
                             "premium": (2000, 5000), "extended": (3000, 10000)}

            min_prem, max_prem = premium_ranges[policy_type]
            min_cov, max_cov = coverage_ranges[policy_type]

            self.policies.append({
                "policy_id": self._next_id("POL"),
                "customer_id": cust_id,
                "device_id": dev_id,
                "policy_type": policy_type,
                "start_date": str(start),
                "end_date": str(end),
                "premium_amount": round(float(self.rng.uniform(min_prem, max_prem)), 2),
                "coverage_limit": round(float(self.rng.uniform(min_cov, max_cov)), 2),
                "is_active": bool(end >= date.today())
            })

    def generate_claims(self) -> None:
        """
        Claims generate karo — har claim ek policy se linked.
        Ye CLEAN claims hain — fraud rings baad mein inject honge.
        """
        print(f"📝 Generating {self.n_claims} clean claims...")

        policy_data = [(p["policy_id"], p["customer_id"], p["device_id"]) for p in self.policies]
        repair_shop_ids = [r["repair_shop_id"] for r in self.repair_shops]
        claim_types = list(CLAIM_DESCRIPTIONS.keys())

        for _ in range(self.n_claims):
            pol_id, cust_id, dev_id = self.rng.choice(policy_data)
            claim_type = self.rng.choice(claim_types)

            # Claim amount — mostly reasonable, kuch high (legitimate outliers)
            base_amount = float(self.rng.lognormal(5.5, 0.8))  # Lognormal = realistic
            claim_amount = min(round(base_amount, 2), 15000.0)  # Cap at 15k

            claim_date = random_date(date(2023, 1, 1), date(2024, 12, 1), self.rng)

            # 60% claims mein repair shop involved
            repair_id = self.rng.choice(repair_shop_ids) if self.rng.random() < 0.6 else None

            # Description choose karo
            descriptions = CLAIM_DESCRIPTIONS[claim_type]
            desc = self.rng.choice(descriptions)
            if "{num}" in desc:
                desc = desc.replace("{num}", str(self.rng.integers(10000, 99999)))

            self.claims.append({
                "claim_id": self._next_id("CLM"),
                "policy_id": pol_id,
                "customer_id": cust_id,
                "device_id": dev_id,
                "repair_shop_id": repair_id,
                "claim_type": claim_type,
                "claim_amount": claim_amount,
                "claim_date": str(claim_date),
                "description": desc,
                "status": self.rng.choice(["submitted", "under_review", "approved", "denied"]),
                "filed_date": str(datetime.now()),
                "image_path": None
            })

    # =============================================================
    # STEP 2: Inject Fraud Rings (the core differentiator)
    # =============================================================

    def inject_fraud_rings(self, n_rings: int = 20) -> None:
        """
        Fraud rings inject karo — ye project ka CORE hai.

        Fraud ring types:
        1. Shared Device Ring: Same device, alag alag "customers"
        2. Shared Repair Shop Ring: Ek shop, unusually high-value claims cluster
        3. Shared Payment Ring: Different customers, same payment account
        4. Mixed Ring: Multiple sharing patterns combined

        Interview answer: "I inject rings by deliberately reusing entities across
        a small cluster of claims. Each ring has 3-6 members. The clean population
        was generated first with independent relationships, so these shared entities
        stand out as anomalous patterns — which is exactly what the graph algorithms
        should detect."
        """
        print(f"🚨 Injecting {n_rings} fraud rings...")

        ring_types = ["shared_device", "shared_shop", "shared_payment", "mixed"]

        for ring_idx in range(n_rings):
            ring_id = self._next_id("RING")
            ring_type = ring_types[ring_idx % len(ring_types)]
            ring_size = int(self.rng.integers(3, 7))  # 3-6 members per ring

            # Ring ke claims ek tight time window mein file hote hain (coordinated)
            base_date = random_date(date(2023, 6, 1), date(2024, 10, 1), self.rng)
            # Time window: 3-14 days mein saare claims
            time_window_days = int(self.rng.integers(3, 15))

            if ring_type == "shared_device":
                self._inject_shared_device_ring(ring_id, ring_size, base_date, time_window_days)
            elif ring_type == "shared_shop":
                self._inject_shared_shop_ring(ring_id, ring_size, base_date, time_window_days)
            elif ring_type == "shared_payment":
                self._inject_shared_payment_ring(ring_id, ring_size, base_date, time_window_days)
            else:  # mixed
                self._inject_mixed_ring(ring_id, ring_size, base_date, time_window_days)

        print(f"   ✅ Injected {n_rings} rings with {len(self.fraud_ground_truth)} fraudulent claims")

    def _inject_shared_device_ring(self, ring_id, ring_size, base_date, window_days):
        """
        Shared Device Ring: Same device claimed by multiple "different" customers.
        Red flag: Ek device 5 alag customers ke paas kaise?
        """
        # Pick one device to share across the ring
        shared_device = self.rng.choice(self.devices)
        repair_shop = self.rng.choice(self.repair_shops)

        for i in range(ring_size):
            customer = self.rng.choice(self.customers)
            policy = self.rng.choice(self.policies)
            claim_date = base_date + timedelta(days=int(self.rng.integers(0, window_days)))

            # Higher than average claim amounts (fraud signal)
            claim_amount = round(float(self.rng.uniform(800, 5000)), 2)

            claim_id = self._next_id("CLM")
            self.claims.append({
                "claim_id": claim_id,
                "policy_id": policy["policy_id"],
                "customer_id": customer["customer_id"],
                "device_id": shared_device["device_id"],  # SHARED device
                "repair_shop_id": repair_shop["repair_shop_id"],
                "claim_type": self.rng.choice(list(CLAIM_DESCRIPTIONS.keys())),
                "claim_amount": claim_amount,
                "claim_date": str(claim_date),
                "description": "Device showing signs of damage, requesting repair coverage",
                "status": "submitted",
                "filed_date": str(datetime.now()),
                "image_path": None
            })

            self.fraud_claim_ids.add(claim_id)
            self.fraud_ground_truth.append({
                "claim_id": claim_id,
                "fraud_ring_id": ring_id,
                "ring_type": "shared_device",
                "is_fraud": True
            })

    def _inject_shared_shop_ring(self, ring_id, ring_size, base_date, window_days):
        """
        Shared Repair Shop Ring: Ek shop se unusually high-value claims ka cluster.
        Red flag: Same shop, tight time window, inflated amounts.
        """
        shared_shop = self.rng.choice(self.repair_shops)

        for i in range(ring_size):
            customer = self.rng.choice(self.customers)
            policy = self.rng.choice(self.policies)
            device = self.rng.choice(self.devices)
            claim_date = base_date + timedelta(days=int(self.rng.integers(0, window_days)))

            # Inflated amounts — fraud signal
            claim_amount = round(float(self.rng.uniform(2000, 8000)), 2)

            claim_id = self._next_id("CLM")
            self.claims.append({
                "claim_id": claim_id,
                "policy_id": policy["policy_id"],
                "customer_id": customer["customer_id"],
                "device_id": device["device_id"],
                "repair_shop_id": shared_shop["repair_shop_id"],  # SHARED shop
                "claim_type": "mechanical_failure",
                "claim_amount": claim_amount,
                "claim_date": str(claim_date),
                "description": "Device requires extensive repair, major component failure",
                "status": "submitted",
                "filed_date": str(datetime.now()),
                "image_path": None
            })

            self.fraud_claim_ids.add(claim_id)
            self.fraud_ground_truth.append({
                "claim_id": claim_id,
                "fraud_ring_id": ring_id,
                "ring_type": "shared_shop",
                "is_fraud": True
            })

    def _inject_shared_payment_ring(self, ring_id, ring_size, base_date, window_days):
        """
        Shared Payment Ring: Different "customers" using same payment account.
        Red flag: Alag customers ka same bank account = likely same person.
        """
        shared_payment = self.rng.choice(self.payment_accounts)

        for i in range(ring_size):
            # Pick customer but override their payment to shared one
            customer = self.rng.choice(self.customers)
            policy = self.rng.choice(self.policies)
            device = self.rng.choice(self.devices)
            claim_date = base_date + timedelta(days=int(self.rng.integers(0, window_days)))

            claim_amount = round(float(self.rng.uniform(500, 4000)), 2)
            claim_id = self._next_id("CLM")

            self.claims.append({
                "claim_id": claim_id,
                "policy_id": policy["policy_id"],
                "customer_id": customer["customer_id"],
                "device_id": device["device_id"],
                "repair_shop_id": self.rng.choice(self.repair_shops)["repair_shop_id"],
                "claim_type": self.rng.choice(list(CLAIM_DESCRIPTIONS.keys())),
                "claim_amount": claim_amount,
                "claim_date": str(claim_date),
                "description": "Filing claim for device damage, need replacement",
                "status": "submitted",
                "filed_date": str(datetime.now()),
                "image_path": None
            })

            # Customer ko shared payment assign karo
            for c in self.customers:
                if c["customer_id"] == customer["customer_id"]:
                    c["payment_account_id"] = shared_payment["payment_account_id"]
                    break

            self.fraud_claim_ids.add(claim_id)
            self.fraud_ground_truth.append({
                "claim_id": claim_id,
                "fraud_ring_id": ring_id,
                "ring_type": "shared_payment",
                "is_fraud": True
            })

    def _inject_mixed_ring(self, ring_id, ring_size, base_date, window_days):
        """
        Mixed Ring: Multiple sharing patterns combined — hardest to detect individually.
        Shared device + shared shop + tight timing + high amounts.
        """
        shared_device = self.rng.choice(self.devices)
        shared_shop = self.rng.choice(self.repair_shops)
        shared_payment = self.rng.choice(self.payment_accounts)

        for i in range(ring_size):
            customer = self.rng.choice(self.customers)
            policy = self.rng.choice(self.policies)
            claim_date = base_date + timedelta(days=int(self.rng.integers(0, window_days)))

            claim_amount = round(float(self.rng.uniform(1500, 7000)), 2)
            claim_id = self._next_id("CLM")

            self.claims.append({
                "claim_id": claim_id,
                "policy_id": policy["policy_id"],
                "customer_id": customer["customer_id"],
                "device_id": shared_device["device_id"],
                "repair_shop_id": shared_shop["repair_shop_id"],
                "claim_type": self.rng.choice(["accidental_damage", "mechanical_failure"]),
                "claim_amount": claim_amount,
                "claim_date": str(claim_date),
                "description": "Device malfunctioned and needs comprehensive repair",
                "status": "submitted",
                "filed_date": str(datetime.now()),
                "image_path": None
            })

            # Shared payment bhi assign karo
            for c in self.customers:
                if c["customer_id"] == customer["customer_id"]:
                    c["payment_account_id"] = shared_payment["payment_account_id"]
                    break

            self.fraud_claim_ids.add(claim_id)
            self.fraud_ground_truth.append({
                "claim_id": claim_id,
                "fraud_ring_id": ring_id,
                "ring_type": "mixed",
                "is_fraud": True
            })

    # =============================================================
    # STEP 3: Ensure Demo Claim Exists
    # =============================================================

    def ensure_demo_claim(self):
        """
        CLM-DEMO-001 create karo — demo walkthrough ke liye guaranteed fraud claim.
        Ye ek known fraud ring ka part hai, high risk hona chahiye.
        """
        print("🎯 Creating demo claim CLM-DEMO-001...")

        # Demo ring — mixed type, clearly fraudulent
        ring_id = self._next_id("RING")
        shared_device = self.devices[0]
        shared_shop = self.repair_shops[0]

        demo_customers = self.customers[:4]
        base_date = date(2024, 8, 15)

        for i, cust in enumerate(demo_customers):
            if i == 0:
                claim_id = "CLM-DEMO-001"  # Special demo ID
            else:
                claim_id = self._next_id("CLM")

            policy = self.rng.choice(self.policies)
            claim_date = base_date + timedelta(days=i)

            self.claims.append({
                "claim_id": claim_id,
                "policy_id": policy["policy_id"],
                "customer_id": cust["customer_id"],
                "device_id": shared_device["device_id"],
                "repair_shop_id": shared_shop["repair_shop_id"],
                "claim_type": "accidental_damage",
                "claim_amount": round(float(self.rng.uniform(2500, 5000)), 2),
                "claim_date": str(claim_date),
                "description": "Front bumper damage on device, screen completely shattered with visible cracks",
                "status": "submitted",
                "filed_date": str(datetime.now()),
                "image_path": "data/images/demo_claim_001.png"
            })

            self.fraud_claim_ids.add(claim_id)
            self.fraud_ground_truth.append({
                "claim_id": claim_id,
                "fraud_ring_id": ring_id,
                "ring_type": "mixed",
                "is_fraud": True
            })

    # =============================================================
    # STEP 4: Add is_fraud label to claims (for training)
    # =============================================================

    def add_fraud_labels(self):
        """
        Claims mein is_fraud column add karo — ye TRAINING ke liye hai.
        NOTE: fraud_ring_id ya ring_type KABHI features mein nahi jaata.
        Sirf binary is_fraud label training mein use hota hai.

        Leakage prevention:
        - is_fraud = binary label (0/1) — ye supervised learning ke liye zaroori hai
        - fraud_ring_id = ground truth ring membership — ALAG file, KABHI features mein nahi
        """
        for claim in self.claims:
            claim["is_fraud"] = claim["claim_id"] in self.fraud_claim_ids

    # =============================================================
    # STEP 5: Save Everything
    # =============================================================

    def save_all(self, output_dir: str) -> dict:
        """Saara data CSV files mein save karo"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Saving data to {output_dir}/...")

        datasets = {
            "customers": self.customers,
            "addresses": self.addresses,
            "devices": self.devices,
            "payment_accounts": self.payment_accounts,
            "repair_shops": self.repair_shops,
            "policies": self.policies,
            "claims": self.claims,
        }

        stats = {}
        for name, data in datasets.items():
            df = pd.DataFrame(data)
            filepath = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            stats[name] = len(df)
            print(f"   ✅ {name}.csv — {len(df)} records")

        # Ground truth ALAG file mein — leakage prevention
        gt_df = pd.DataFrame(self.fraud_ground_truth)
        gt_path = os.path.join(output_dir, "fraud_ring_ground_truth.csv")
        gt_df.to_csv(gt_path, index=False)
        stats["fraud_rings"] = gt_df["fraud_ring_id"].nunique()
        stats["fraud_claims"] = len(gt_df)
        print(f"   ✅ fraud_ring_ground_truth.csv — {len(gt_df)} records ({stats['fraud_rings']} rings)")

        # Generation report — metadata
        report = {
            "generated_at": str(datetime.now()),
            "seed": self.seed,
            "params": {
                "n_customers": self.n_customers,
                "n_policies": self.n_policies,
                "n_claims": self.n_claims
            },
            "stats": stats,
            "fraud_rate": round(stats["fraud_claims"] / stats["claims"] * 100, 2)
        }

        report_path = os.path.join(output_dir, "data_generation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"   ✅ data_generation_report.json")

        return report

    # =============================================================
    # MAIN: Full Pipeline
    # =============================================================

    def generate_all(self) -> dict:
        """
        Full data generation pipeline:
        1. Clean population (independent entities)
        2. Fraud ring injection (deliberate entity sharing)
        3. Demo claim creation
        4. Fraud labels
        5. Save everything
        """
        print("=" * 60)
        print("🏗️  ClaimGraph AI — Synthetic Data Generation")
        print("=" * 60)
        print(f"   Customers: {self.n_customers}")
        print(f"   Policies:  {self.n_policies}")
        print(f"   Claims:    {self.n_claims}")
        print(f"   Seed:      {self.seed}")
        print("=" * 60)

        # Step 1: Clean population
        print("\n📊 Step 1: Generating clean population...")
        self.generate_addresses()
        self.generate_payment_accounts()
        self.generate_repair_shops()
        self.generate_devices()
        self.generate_customers()
        self.generate_policies()
        self.generate_claims()

        # Step 2: Fraud ring injection
        print("\n🚨 Step 2: Injecting fraud rings...")
        self.inject_fraud_rings(n_rings=20)

        # Step 3: Demo claim
        print("\n🎯 Step 3: Creating demo claim...")
        self.ensure_demo_claim()

        # Step 4: Labels
        print("\n🏷️  Step 4: Adding fraud labels...")
        self.add_fraud_labels()

        # Step 5: Save
        output_dir = str(Path(__file__).parent.parent / "data" / "generated")
        report = self.save_all(output_dir)

        print("\n" + "=" * 60)
        print("✅ Data generation complete!")
        print(f"   Total claims: {report['stats']['claims']}")
        print(f"   Fraud claims: {report['stats']['fraud_claims']}")
        print(f"   Fraud rate:   {report['fraud_rate']}%")
        print(f"   Fraud rings:  {report['stats']['fraud_rings']}")
        print("=" * 60)

        return report


# === CLI Entry Point ===

def main():
    parser = argparse.ArgumentParser(
        description="ClaimGraph AI — Synthetic Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_data.py --customers 8000 --policies 15000 --claims 25000 --seed 42
  python scripts/generate_data.py  # Uses defaults
        """
    )
    parser.add_argument("--customers", type=int, default=8000,
                       help="Number of customers (default: 8000)")
    parser.add_argument("--policies", type=int, default=15000,
                       help="Number of policies (default: 15000)")
    parser.add_argument("--claims", type=int, default=25000,
                       help="Number of claims (default: 25000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    generator = SyntheticDataGenerator(
        n_customers=args.customers,
        n_policies=args.policies,
        n_claims=args.claims,
        seed=args.seed
    )

    generator.generate_all()


if __name__ == "__main__":
    main()
