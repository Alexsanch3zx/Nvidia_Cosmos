--Command: psql postgres -f db/schema.sql
--Verify: psql cosmos_videos
--Run: \d video_summaries
--Then you should see: embedding | vector(384)
-- create database
CREATE DATABASE cosmos_videos;

-- connect to DB
\c cosmos_videos

-- enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- create table for video summaries
CREATE TABLE video_summaries (
  id            BIGSERIAL PRIMARY KEY,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  filename      TEXT,
  duration_sec  NUMERIC(10,2),
  summary_style TEXT,
  summary_text  TEXT NOT NULL,
  embedding     vector(384)  -- 384 for all-MiniLM-L6-v2; change if you use another model
);

--Test row
INSERT INTO video_summaries
(filename, duration_sec, summary_style, summary_text, embedding)
VALUES
('test_video.mp4', 60.5, 'concise', 'Test summary', '[0.1,0.2,0.3]');

--index for similarity search speed
--CREATE INDEX ON video_summaries
--USING ivfflat (embedding vector_cosine_ops)
--WITH (lists = 100);