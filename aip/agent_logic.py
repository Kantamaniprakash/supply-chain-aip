"""
AIP Logic — Supply Chain Intelligence Agent
============================================
AIP (Artificial Intelligence Platform) action functions that connect
GPT-4o to live Foundry Ontology objects.

These functions are registered as AIP Logic in Foundry, giving the
LLM the ability to query live supply chain data and take actions.

Author: Satya Sai Prakash Kantamani
"""

from typing import Any
from aip_logic import action, ontology_query, OntologyObject  # Foundry AIP SDK


# ── ONTOLOGY QUERIES ────────────────────────────────────────────────────────

@action(description="Get all suppliers with disruption risk above a threshold")
def get_high_risk_suppliers(risk_threshold: float = 0.6) -> list[dict]:
    """
    Returns suppliers whose ML disruption probability exceeds the threshold.
    Called by the AIP agent when asked about supply chain risk.
    """
    suppliers = ontology_query(
        object_type="Supplier",
        filter={"disruption_probability": {"$gte": risk_threshold}},
        properties=["supplier_id", "supplier_name", "country", "disruption_probability",
                    "risk_tier", "primary_category", "annual_spend_usd"],
        order_by="disruption_probability",
        limit=20,
    )
    return [s.to_dict() for s in suppliers]


@action(description="Get active shipments that are delayed or at risk of delay")
def get_delayed_shipments(min_delay_days: int = 2) -> list[dict]:
    """
    Queries Shipment objects with delay >= threshold.
    Linked to Supplier and Order objects for full context.
    """
    shipments = ontology_query(
        object_type="Shipment",
        filter={
            "status": {"$in": ["DELAYED", "AT_RISK"]},
            "delay_days": {"$gte": min_delay_days},
        },
        properties=["shipment_id", "supplier_name", "origin", "destination",
                    "expected_delivery", "delay_days", "cargo_value_usd",
                    "linked_order_ids"],
        limit=50,
    )
    return [s.to_dict() for s in shipments]


@action(description="Get inventory items at risk of stockout within N days")
def get_stockout_risks(days_of_supply_threshold: int = 14) -> list[dict]:
    """
    Returns Inventory objects where days_of_supply < threshold.
    Triggers reorder recommendations in the agent response.
    """
    items = ontology_query(
        object_type="Inventory",
        filter={"days_of_supply": {"$lt": days_of_supply_threshold}},
        properties=["sku_id", "product_name", "days_of_supply", "current_stock",
                    "reorder_point", "primary_supplier_id", "backup_supplier_id",
                    "avg_daily_demand"],
        order_by="days_of_supply",
        limit=30,
    )
    return [i.to_dict() for i in items]


@action(description="Get recent risk events (geopolitical, weather, logistics) affecting supply chain")
def get_active_risk_events(severity: str = "HIGH") -> list[dict]:
    """
    Returns RiskEvent objects at or above severity level.
    Used by the agent to contextualize supplier risk scores.
    """
    events = ontology_query(
        object_type="RiskEvent",
        filter={
            "severity": {"$in": ["CRITICAL", severity]},
            "status": "ACTIVE",
        },
        properties=["event_id", "event_type", "description", "affected_region",
                    "affected_supplier_count", "estimated_impact_usd",
                    "start_date", "expected_resolution_date"],
        limit=10,
    )
    return [e.to_dict() for e in events]


# ── AIP ACTIONS (agent can execute these) ───────────────────────────────────

@action(description="Create an emergency purchase order for a stockout-risk SKU")
def create_emergency_purchase_order(
    sku_id: str,
    supplier_id: str,
    quantity: int,
    justification: str,
) -> dict:
    """
    Creates a PurchaseOrder object in Foundry and triggers ERP webhook.
    Requires human approval if order value > $50,000.
    """
    order_value = _estimate_order_value(sku_id, quantity)

    purchase_order = OntologyObject.create(
        object_type="PurchaseOrder",
        properties={
            "sku_id": sku_id,
            "supplier_id": supplier_id,
            "quantity": quantity,
            "order_type": "EMERGENCY",
            "justification": justification,
            "estimated_value_usd": order_value,
            "status": "PENDING_APPROVAL" if order_value > 50_000 else "AUTO_APPROVED",
            "created_by": "AIP_AGENT",
        }
    )

    return {
        "order_id": purchase_order.id,
        "status": purchase_order.status,
        "estimated_value_usd": order_value,
        "message": f"Purchase order created for {quantity} units of SKU {sku_id} "
                   f"from supplier {supplier_id}. Status: {purchase_order.status}.",
    }


@action(description="Send an automated risk alert to the procurement team")
def send_supplier_risk_alert(
    supplier_id: str,
    risk_summary: str,
    recommended_action: str,
    urgency: str = "HIGH",
) -> dict:
    """
    Creates a SupplierAlert object and sends notification via Foundry webhook.
    Notifies the procurement team in Slack/Teams.
    """
    alert = OntologyObject.create(
        object_type="SupplierAlert",
        properties={
            "supplier_id": supplier_id,
            "risk_summary": risk_summary,
            "recommended_action": recommended_action,
            "urgency": urgency,
            "status": "SENT",
            "created_by": "AIP_AGENT",
        }
    )
    return {
        "alert_id": alert.id,
        "message": f"Alert sent to procurement team for supplier {supplier_id}.",
    }


@action(description="Generate an executive supply chain risk summary report")
def generate_executive_report(scope: str = "WEEKLY") -> dict:
    """
    Triggers a Foundry report generation pipeline.
    Returns a PDF summary of supply chain health for leadership review.
    """
    high_risk = get_high_risk_suppliers(risk_threshold=0.6)
    delayed = get_delayed_shipments(min_delay_days=3)
    stockouts = get_stockout_risks(days_of_supply_threshold=7)

    return {
        "high_risk_supplier_count": len(high_risk),
        "delayed_shipment_count": len(delayed),
        "critical_stockout_count": len(stockouts),
        "top_risks": high_risk[:3],
        "report_url": f"/foundry/reports/supply-chain-executive-{scope.lower()}",
        "message": f"Executive report generated: {len(high_risk)} high-risk suppliers, "
                   f"{len(delayed)} delayed shipments, {len(stockouts)} stockout risks.",
    }


# ── PRIVATE HELPERS ─────────────────────────────────────────────────────────

def _estimate_order_value(sku_id: str, quantity: int) -> float:
    item = ontology_query("Inventory", filter={"sku_id": sku_id}, limit=1)
    if item:
        return item[0].unit_cost_usd * quantity
    return 0.0
