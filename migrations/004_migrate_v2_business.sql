-- 将 siglip_v2 业务表并入 medical_audit_v3。
-- 本文件只改 v3 结构，不删除、不修改 siglip_v2。
-- 不可重复执行：第二次 ALTER ADD COLUMN / ADD KEY 会失败。
-- 上线请用幂等脚本: python migrations/backfill_v2_to_v3.py

USE medical_audit_v3;

-- videos：承接 processed_files 的任务状态，用 id 与轨迹/房间正式关联
ALTER TABLE videos
  ADD COLUMN origin_label VARCHAR(50) NOT NULL DEFAULT '手动上传'
    COMMENT '前端展示用来源：手动上传/NVR流/NVR_ISAPI/RTSP实时' AFTER source_type,
  ADD COLUMN task_id VARCHAR(100) NULL COMMENT 'Celery 任务 ID' AFTER origin_label,
  ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT 'pending'
    COMMENT 'pending/processing/transcoding/completed/failed/streaming/stopped' AFTER task_id,
  ADD COLUMN failure_reason VARCHAR(512) NULL AFTER status,
  ADD COLUMN progress INT NOT NULL DEFAULT 0 AFTER failure_reason,
  ADD COLUMN duration VARCHAR(32) NULL COMMENT '展示用时长 HH:MM:SS 或直播文案' AFTER duration_sec,
  ADD COLUMN target_count INT NOT NULL DEFAULT 0 AFTER duration,
  ADD COLUMN completed_at DATETIME(3) NULL AFTER target_count,
  ADD COLUMN processing_started_at DATETIME(3) NULL AFTER completed_at,
  ADD COLUMN updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
    ON UPDATE CURRENT_TIMESTAMP(3) AFTER created_at,
  ADD COLUMN is_media_source TINYINT(1) NOT NULL DEFAULT 0
    COMMENT '1=媒体源列表中的任务视频；0=仅轨迹实验数据' AFTER source_path;

ALTER TABLE videos
  ADD UNIQUE KEY uq_videos_file_name (file_name),
  ADD KEY idx_videos_media (is_media_source, created_at),
  ADD KEY idx_videos_task_id (task_id),
  ADD KEY idx_videos_status (status);

-- 房间按 video_id + 名称唯一，替代 video_rooms(video_name, room_name)
ALTER TABLE rooms
  ADD UNIQUE KEY uq_rooms_video_name (video_id, name);

CREATE TABLE IF NOT EXISTS users (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  username VARCHAR(64) NOT NULL COMMENT '登录名',
  password_hash VARCHAR(255) NOT NULL COMMENT 'bcrypt 哈希',
  role VARCHAR(20) NOT NULL DEFAULT 'user' COMMENT 'admin / user',
  is_active TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统用户';

CREATE TABLE IF NOT EXISTS search_logs (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  user_id INT UNSIGNED NULL COMMENT '检索用户，旧数据允许为空',
  query_image_paths TEXT COMMENT '查询图片路径（逗号分隔的多个文件名）',
  search_query VARCHAR(500) DEFAULT NULL COMMENT '文字检索描述',
  algorithm VARCHAR(50) NOT NULL COMMENT '检索算法: OSNet/SIGLIP',
  threshold FLOAT NOT NULL DEFAULT 0.82 COMMENT '相似度阈值',
  results_json LONGTEXT COMMENT '检索结果JSON数据',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '检索时间',
  PRIMARY KEY (id),
  KEY idx_created_at (created_at),
  KEY idx_algorithm (algorithm),
  KEY idx_search_logs_user (user_id),
  CONSTRAINT fk_search_logs_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='轨迹检索历史日志';

CREATE TABLE IF NOT EXISTS gallery_meta (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  video_id BIGINT UNSIGNED NULL COMMENT '所属 videos.id',
  video_name VARCHAR(255) NOT NULL COMMENT '冗余文件名，便于检索展示',
  timestamp FLOAT NOT NULL COMMENT '时间戳(秒)',
  image_path VARCHAR(500) NOT NULL COMMENT '快照图片路径',
  bbox_x1 INT DEFAULT NULL,
  bbox_y1 INT DEFAULT NULL,
  bbox_x2 INT DEFAULT NULL,
  bbox_y2 INT DEFAULT NULL,
  feature_vector LONGBLOB COMMENT 'OSNet特征向量(512维)',
  clip_feature LONGBLOB COMMENT 'SigLIP特征向量(768维)',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_video_name (video_name),
  KEY idx_timestamp (timestamp),
  KEY idx_gallery_video_id (video_id),
  CONSTRAINT fk_gallery_meta_video
    FOREIGN KEY (video_id) REFERENCES videos(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='行人特征底库（检索）';
