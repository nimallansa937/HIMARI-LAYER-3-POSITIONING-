#!/usr/bin/env python3
"""
HIMARI Layer 3 - Cloud Monitoring Setup
Creates dashboards and alert policies for the RL inference API.
"""

from google.cloud import monitoring_v3
from google.cloud.monitoring_dashboard import v1 as dashboard_v1
import json

PROJECT_ID = "himari-opus-position-layer"
SERVICE_NAME = "himari-rl-api"


def create_alert_policy_latency():
    """Create alert for high latency (>100ms P95)."""
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{PROJECT_ID}"
    
    alert_policy = monitoring_v3.AlertPolicy(
        display_name="HIMARI RL API - High Latency",
        documentation=monitoring_v3.AlertPolicy.Documentation(
            content="P95 latency exceeded 100ms for RL inference API",
            mime_type="text/markdown"
        ),
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="P95 Latency > 100ms",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter=f'resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_latencies" AND resource.labels.service_name="{SERVICE_NAME}"',
                    aggregations=[
                        monitoring_v3.Aggregation(
                            alignment_period={"seconds": 60},
                            per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_PERCENTILE_95
                        )
                    ],
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                    threshold_value=100.0,  # 100ms
                    duration={"seconds": 300}  # 5 minutes
                )
            )
        ],
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
        enabled=True
    )
    
    created = client.create_alert_policy(name=project_name, alert_policy=alert_policy)
    print(f"Created latency alert: {created.name}")
    return created


def create_alert_policy_errors():
    """Create alert for high error rate (>5%)."""
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{PROJECT_ID}"
    
    alert_policy = monitoring_v3.AlertPolicy(
        display_name="HIMARI RL API - High Error Rate",
        documentation=monitoring_v3.AlertPolicy.Documentation(
            content="Error rate exceeded 5% for RL inference API",
            mime_type="text/markdown"
        ),
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="Error Rate > 5%",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter=f'resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count" AND metric.labels.response_code_class!="2xx" AND resource.labels.service_name="{SERVICE_NAME}"',
                    aggregations=[
                        monitoring_v3.Aggregation(
                            alignment_period={"seconds": 60},
                            per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_RATE
                        )
                    ],
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                    threshold_value=0.05,  # 5%
                    duration={"seconds": 300}
                )
            )
        ],
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
        enabled=True
    )
    
    created = client.create_alert_policy(name=project_name, alert_policy=alert_policy)
    print(f"Created error rate alert: {created.name}")
    return created


def create_dashboard():
    """Create monitoring dashboard for RL API."""
    client = dashboard_v1.DashboardsServiceClient()
    
    dashboard = dashboard_v1.Dashboard(
        display_name="HIMARI RL Inference API",
        grid_layout=dashboard_v1.GridLayout(
            columns=2,
            widgets=[
                # Request Count
                dashboard_v1.Widget(
                    title="Request Count",
                    xy_chart=dashboard_v1.XyChart(
                        data_sets=[
                            dashboard_v1.XyChart.DataSet(
                                time_series_query=dashboard_v1.TimeSeriesQuery(
                                    time_series_filter=dashboard_v1.TimeSeriesFilter(
                                        filter=f'resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count" AND resource.labels.service_name="{SERVICE_NAME}"'
                                    )
                                )
                            )
                        ]
                    )
                ),
                # Latency
                dashboard_v1.Widget(
                    title="Request Latency (P50, P95, P99)",
                    xy_chart=dashboard_v1.XyChart(
                        data_sets=[
                            dashboard_v1.XyChart.DataSet(
                                time_series_query=dashboard_v1.TimeSeriesQuery(
                                    time_series_filter=dashboard_v1.TimeSeriesFilter(
                                        filter=f'resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_latencies" AND resource.labels.service_name="{SERVICE_NAME}"'
                                    )
                                )
                            )
                        ]
                    )
                ),
                # Container Instances
                dashboard_v1.Widget(
                    title="Active Instances",
                    xy_chart=dashboard_v1.XyChart(
                        data_sets=[
                            dashboard_v1.XyChart.DataSet(
                                time_series_query=dashboard_v1.TimeSeriesQuery(
                                    time_series_filter=dashboard_v1.TimeSeriesFilter(
                                        filter=f'resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/container/instance_count" AND resource.labels.service_name="{SERVICE_NAME}"'
                                    )
                                )
                            )
                        ]
                    )
                ),
                # Memory Usage
                dashboard_v1.Widget(
                    title="Memory Usage",
                    xy_chart=dashboard_v1.XyChart(
                        data_sets=[
                            dashboard_v1.XyChart.DataSet(
                                time_series_query=dashboard_v1.TimeSeriesQuery(
                                    time_series_filter=dashboard_v1.TimeSeriesFilter(
                                        filter=f'resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/container/memory/utilizations" AND resource.labels.service_name="{SERVICE_NAME}"'
                                    )
                                )
                            )
                        ]
                    )
                )
            ]
        )
    )
    
    created = client.create_dashboard(
        parent=f"projects/{PROJECT_ID}",
        dashboard=dashboard
    )
    print(f"Created dashboard: {created.name}")
    return created


def main():
    """Setup all monitoring resources."""
    print("="*60)
    print("HIMARI Layer 3 - Cloud Monitoring Setup")
    print("="*60)
    print(f"Project: {PROJECT_ID}")
    print(f"Service: {SERVICE_NAME}")
    print()
    
    try:
        print("Creating alert policies...")
        create_alert_policy_latency()
        create_alert_policy_errors()
        
        print("\nCreating dashboard...")
        create_dashboard()
        
        print("\n" + "="*60)
        print("✓ Monitoring setup complete!")
        print("="*60)
        print(f"\nView dashboard at:")
        print(f"https://console.cloud.google.com/monitoring/dashboards?project={PROJECT_ID}")
        print(f"\nView alerts at:")
        print(f"https://console.cloud.google.com/monitoring/alerting?project={PROJECT_ID}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have the Cloud Monitoring API enabled")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
