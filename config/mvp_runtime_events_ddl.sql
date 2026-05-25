-- public.runtime_run definition

-- Drop table

-- DROP TABLE public.runtime_run;

CREATE TABLE public.runtime_run (
	run_id uuid NOT NULL,
	template_id uuid NULL,
	template_version text NULL,
	graph_hash text NULL,
	plan_hash text NULL,
	input_asset_id uuid NULL,
	status text NULL,
	started_at timestamptz NULL,
	finished_at timestamptz NULL,
	total_latency_ms int4 NULL,
	created_at timestamptz DEFAULT now() NULL,
	CONSTRAINT runtime_run_pkey PRIMARY KEY (run_id)
);


-- public.runtime_plan_snapshot definition

-- Drop table

-- DROP TABLE public.runtime_plan_snapshot;

CREATE TABLE public.runtime_plan_snapshot (
	snapshot_id uuid NOT NULL,
	run_id uuid NULL,
	graph_json jsonb NULL,
	graph_hash text NULL,
	created_at timestamptz DEFAULT now() NULL,
	CONSTRAINT runtime_plan_snapshot_pkey PRIMARY KEY (snapshot_id),
	CONSTRAINT runtime_plan_snapshot_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.runtime_run(run_id)
);


-- public.runtime_step definition

-- Drop table

-- DROP TABLE public.runtime_step;

CREATE TABLE public.runtime_step (
	step_id uuid NOT NULL,
	run_id uuid NULL,
	step_name text NULL,
	step_type text NULL,
	"handler" text NULL,
	model_id text NULL,
	status text NULL,
	started_at timestamptz NULL,
	finished_at timestamptz NULL,
	latency_ms int4 NULL,
	input_candidate_count int4 NULL,
	output_candidate_count int4 NULL,
	error_message text NULL,
	created_at timestamptz DEFAULT now() NULL,
	CONSTRAINT runtime_step_pkey PRIMARY KEY (step_id),
	CONSTRAINT runtime_step_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.runtime_run(run_id)
);


-- public.runtime_step_event definition

-- Drop table

-- DROP TABLE public.runtime_step_event;

CREATE TABLE public.runtime_step_event (
	event_id uuid NOT NULL,
	step_id uuid NULL,
	event_type text NULL,
	payload jsonb NULL,
	created_at timestamptz DEFAULT now() NULL,
	CONSTRAINT runtime_step_event_pkey PRIMARY KEY (event_id),
	CONSTRAINT runtime_step_event_step_id_fkey FOREIGN KEY (step_id) REFERENCES public.runtime_step(step_id)
);