-- 分析报告软删除。幂等由 apply_007 检查 information_schema。
USE medical_audit_v3;

ALTER TABLE analysis_reports
  ADD COLUMN deleted_at DATETIME(3) NULL COMMENT '软删除时间（UTC naive）' AFTER updated_at,
  ADD COLUMN deleted_by INT UNSIGNED NULL COMMENT '删除人 users.id' AFTER deleted_at;

ALTER TABLE analysis_reports
  ADD INDEX idx_analysis_reports_deleted (deleted_at);

ALTER TABLE analysis_reports
  ADD CONSTRAINT fk_analysis_reports_deleted_by
    FOREIGN KEY (deleted_by) REFERENCES users(id)
    ON DELETE RESTRICT;
