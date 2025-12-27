# monitoring/setup_monitoring.py
"""
Setup Cloud Monitoring for HIMARI RL system.
"""

from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import query
import time

PROJECT_ID = "himari-opus-2"
PROJECT_NAME = f"projects/{PROJECT_ID}"


def create_custom_metrics():
    """Create custom metrics for HIMARI RL."""

    client = monitoring_v3.MetricServiceClient()

    metrics = [
        {
            "type": "custom.googleapis.com/himari/rl/prediction_latency",
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
            "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            "description": "RL API prediction latency in milliseconds",
            "unit": "ms"
        },
        {
            "type": "custom.googleapis.com/himari/rl/position_multiplier",
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
            "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            "description": "RL position size multiplier [0.0-2.0]",
            "unit": "1"
        },
        {
            "type": "custom.googleapis.com/himari/rl/api_errors",
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.CUMULATIVE,
            "value_type": monitoring_v3.MetricDescriptor.ValueType.INT64,
            "description": "Count of RL API errors",
            "unit": "1"
        },
        {
            "type": "custom.googleapis.com/himari/rl/fallback_count",
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.CUMULATIVE,
            "value_type": monitoring_v3.MetricDescriptor.ValueType.INT64,
            "description": "Count of fallbacks to Bayesian Kelly",
            "unit": "1"
        },
    ]

    for metric_def in metrics:
        descriptor = monitoring_v3.MetricDescriptor(
            type_=metric_def["type"],
            metric_kind=metric_def["metric_kind"],
            value_type=metric_def["value_type"],
            description=metric_def["description"],
            unit=metric_def.get("unit", "1")
        )

        try:
            created = client.create_metric_descriptor(
                name=PROJECT_NAME,
                metric_descriptor=descriptor
            )
            print(f"✓ Created metric: {metric_def['type']}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"⊙ Metric already exists: {metric_def['type']}")
            else:
                print(f"✗ Failed to create metric: {e}")


def create_alert_policies():
    """Create alert policies for critical issues."""

    client = monitoring_v3.AlertPolicyServiceClient()

    # Alert: High latency
    high_latency_policy = monitoring_v3.AlertPolicy(
        display_name="HIMARI RL - High Latency",
        documentation=monitoring_v3.AlertPolicy.Documentation(
            content="RL API latency exceeded 150ms threshold. Investigate Cloud Run performance.",
            mime_type="text/markdown"
        ),
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="Latency > 150ms",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='metric.type="custom.googleapis.com/himari/rl/prediction_latency" resource.type="cloud_run_revision"',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                    threshold_value=150.0,
                    duration={"seconds": 60},
                    aggregations=[
                        monitoring_v3.Aggregation(
                            alignment_period={"seconds": 60},
                            per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                        )
                    ],
                )
            )
        ],
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.AND,
        enabled=True,
    )

    # Alert: High error rate
    high_error_policy = monitoring_v3.AlertPolicy(
        display_name="HIMARI RL - High Error Rate",
        documentation=monitoring_v3.AlertPolicy.Documentation(
            content="RL API error rate exceeded 5%. Check Cloud Run logs for failures.",
            mime_type="text/markdown"
        ),
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="Error rate > 5%",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='metric.type="run.googleapis.com/request_count" resource.type="cloud_run_revision" metric.label.response_code_class="5xx"',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                    threshold_value=5.0,
                    duration={"seconds": 300},
                    aggregations=[
                        monitoring_v3.Aggregation(
                            alignment_period={"seconds": 60},
                            per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_RATE,
                        )
                    ],
                )
            )
        ],
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.AND,
        enabled=True,
    )

    policies = [high_latency_policy, high_error_policy]

    for policy in policies:
        try:
            created = client.create_alert_policy(
                name=PROJECT_NAME,
                alert_policy=policy
            )
            print(f"✓ Created alert: {policy.display_name}")
        except Exception as e:
            print(f"✗ Failed to create alert: {e}")


if __name__ == "__main__":
    print("Setting up Cloud Monitoring for HIMARI RL...")
    print("")

    print("Creating custom metrics...")
    create_custom_metrics()
    print("")

    print("Creating alert policies...")
    create_alert_policies()
    print("")

    print("✓ Monitoring setup complete!")
    print("")
    print("View dashboards: https://console.cloud.google.com/monitoring")
