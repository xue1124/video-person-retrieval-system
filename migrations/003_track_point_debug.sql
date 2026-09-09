-- 轨迹点补充聚类调试字段：合并前 ID、在线分配相似度、二次合并相似度
USE medical_audit_v3;

ALTER TABLE track_points
  ADD COLUMN pre_merge_id INT NULL
    COMMENT '聚类在线分配 ID（二次合并前）',
  ADD COLUMN assign_score DECIMAL(7,6) NULL
    COMMENT '该检测框并入当前簇的一对一相似度',
  ADD COLUMN merge_score DECIMAL(7,6) NULL
    COMMENT '该簇被二次合并吸收时的一对一相似度';
