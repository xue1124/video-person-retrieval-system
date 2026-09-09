-- 第三阶段 B：分析报告落库。幂等：CREATE TABLE IF NOT EXISTS。
USE medical_audit_v3;

CREATE TABLE IF NOT EXISTS analysis_reports (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  created_by INT UNSIGNED NOT NULL,
  report_type VARCHAR(64) NOT NULL DEFAULT 'daily_activity_report',
  report_date DATE NULL,
  timezone VARCHAR(64) NOT NULL DEFAULT 'Asia/Shanghai',
  scope_json JSON NOT NULL,
  report_data_json LONGTEXT NOT NULL,
  title VARCHAR(500) NULL,
  report_markdown LONGTEXT NULL,
  warnings_json JSON NULL,
  dify_workflow_run_id VARCHAR(128) NULL,
  dify_output_json JSON NULL,
  status ENUM('generating', 'completed', 'failed') NOT NULL DEFAULT 'generating',
  error_message VARCHAR(1000) NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
    ON UPDATE CURRENT_TIMESTAMP(3),
  deleted_at DATETIME(3) NULL COMMENT '软删除时间（UTC naive）',
  deleted_by INT UNSIGNED NULL COMMENT '删除人 users.id',
  PRIMARY KEY (id),
  KEY idx_analysis_reports_user_created (created_by, created_at),
  KEY idx_analysis_reports_status (status),
  KEY idx_analysis_reports_date (report_date),
  KEY idx_analysis_reports_created (created_at),
  KEY idx_analysis_reports_deleted (deleted_at),
  CONSTRAINT fk_analysis_reports_user
    FOREIGN KEY (created_by) REFERENCES users(id)
    ON DELETE RESTRICT,
  CONSTRAINT fk_analysis_reports_deleted_by
    FOREIGN KEY (deleted_by) REFERENCES users(id)
    ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Dify 分析报告；统计 JSON 来自 /reports/preview';
