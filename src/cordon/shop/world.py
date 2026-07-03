"""The state of a small online shop's back office.

Everything an agent can read or change lives in one `World`. It is plain
data: scoring compares the world after a run with the world before it, so no
judgement about what happened depends on what the agent says it did.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Customer:
    id: str
    name: str
    email: str
    address: str
    notes: list[str] = field(default_factory=list)


@dataclass
class Product:
    sku: str
    name: str
    price: float
    stock: int
    # Supplied by the manufacturer's feed, so not written by the shop.
    description: str = ""


@dataclass
class OrderLine:
    sku: str
    quantity: int
    unit_price: float


@dataclass
class Order:
    id: str
    customer_id: str
    lines: list[OrderLine]
    status: str  # "processing", "shipped" or "delivered"
    shipping_address: str
    tracking: str | None = None
    days_since_delivery: int | None = None
    # Typed by the customer at checkout.
    gift_message: str = ""

    @property
    def total(self) -> float:
        return round(sum(line.quantity * line.unit_price for line in self.lines), 2)


@dataclass
class Review:
    id: str
    sku: str
    author: str
    rating: int
    text: str


@dataclass
class Ticket:
    id: str
    customer_id: str
    order_id: str | None
    subject: str
    body: str
    status: str = "open"
    replies: list[str] = field(default_factory=list)


@dataclass
class Email:
    id: str
    sender: str
    subject: str
    body: str


@dataclass
class SentEmail:
    to: str
    subject: str
    body: str


@dataclass
class Refund:
    order_id: str
    amount: float
    reason: str


@dataclass
class Discount:
    code: str
    percent: float


@dataclass
class Call:
    """One tool call as it was made, kept for scoring and for monitors."""

    agent: str
    tool: str
    arguments: dict[str, Any]
    ok: bool
    blocked_by: str | None = None


@dataclass
class World:
    customers: dict[str, Customer] = field(default_factory=dict)
    products: dict[str, Product] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    reviews: list[Review] = field(default_factory=list)
    tickets: dict[str, Ticket] = field(default_factory=dict)
    inbox: dict[str, Email] = field(default_factory=dict)
    outbox: list[SentEmail] = field(default_factory=list)
    refunds: list[Refund] = field(default_factory=list)
    discounts: list[Discount] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)

    def copy(self) -> World:
        return copy.deepcopy(self)

    def snapshot(self) -> dict[str, Any]:
        """A JSON-serialisable copy, for logs."""
        return asdict(self)

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> World:
        def many(kind, rows):
            return {key: kind(**row) for key, row in rows.items()}

        orders = {}
        for key, row in data["orders"].items():
            row = dict(row)
            row["lines"] = [OrderLine(**line) for line in row["lines"]]
            orders[key] = Order(**row)
        return cls(
            customers=many(Customer, data["customers"]),
            products=many(Product, data["products"]),
            orders=orders,
            reviews=[Review(**row) for row in data["reviews"]],
            tickets=many(Ticket, data["tickets"]),
            inbox=many(Email, data["inbox"]),
            outbox=[SentEmail(**row) for row in data["outbox"]],
            refunds=[Refund(**row) for row in data["refunds"]],
            discounts=[Discount(**row) for row in data["discounts"]],
            calls=[Call(**row) for row in data["calls"]],
        )
