"""medical_audit_v3 跟踪入库旁路包。"""

from .sidecar import run_tracking_v3_sidecar, tracking_v3_enabled

__all__ = ["run_tracking_v3_sidecar", "tracking_v3_enabled"]
