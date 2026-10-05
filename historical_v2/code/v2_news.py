"""Auditable regional-event context, without fabricated affected territories.

Known event context annotates a statistical alarm. It does not alter numeric
forecasts or claim that an external emergency is a consumption changepoint.
"""


def contextualize(territory_id, origin_month, statistical_alarm, events):
    """Month-end availability of exact-territory public event anchors.

Source timezone and economic release vintages are not known. Use this only
for retrospective monthly context; never infer a lead time from it.
"""
    available=[]
    for event in events:
        if (territory_id in event["territory_ids"] and
                event["month"] <= origin_month and
                event.get("download_status")=="DOWNLOADED"):
            available.append(event["event_id"])
    same_month=[event["event_id"] for event in events if event["event_id"] in available and event["month"]==origin_month]
    return dict(statistical_alarm=bool(statistical_alarm),
                published_event_context=available,
                same_month_context=same_month,
                news_supported_alarm=bool(statistical_alarm and same_month),
                numeric_forecast_changed=False,
                interpretation="context support, not shock truth/probability or advance proof")
