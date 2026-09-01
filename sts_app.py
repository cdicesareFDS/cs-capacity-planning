import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date as _date
from decimal import Decimal
import numpy as np
from databricks.sql import connect
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

st.set_page_config(page_title="Workforce Analytics", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# COLOR PALETTE (from dataviz skill)
# ============================================================================
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}

ACTIVITY_NAMES = ["Cases", "Meetings", "Calls", "Emails", "RPDs"]
ACTIVITY_COLORS = dict(zip(ACTIVITY_NAMES, CATEGORICAL))

st.title("Workforce Capacity & Activity Analytics")


# ============================================================================
# HELPERS
# ============================================================================

def capacity_status(hours_per_week, group_average, variance_pct):
    """Determine capacity status relative to group average with variance tolerance"""
    if group_average == 0:
        return "On Target", STATUS["good"], ""

    tolerance = group_average * (variance_pct / 100)
    target_min = group_average - tolerance
    target_max = group_average + tolerance

    if hours_per_week < target_min:
        return "Under", STATUS["warning"], ""
    elif hours_per_week > target_max:
        return "Over", STATUS["critical"], ""
    return "On Target", STATUS["good"], ""

# ============================================================================
# DATA LOADING & PREP
# ============================================================================

def get_databricks_connection():
    """Create a Databricks SQL connection using the logged-in user's credentials."""
    import sys
    import os
    from databricks import sql

    # In Databricks App: use the logged-in user's forwarded token
    user_token = st.context.headers.get("x-forwarded-access-token")
    if user_token:
        # Get host directly from environment (avoid Config() interference with SP creds)
        host = os.environ.get("DATABRICKS_HOST", "").replace("https://", "").replace("http://", "").strip("/")
        print(f"[get_databricks_connection] Using user auth token (len={len(user_token)}, starts={user_token[:8]}...)", file=sys.stderr)
        print(f"[get_databricks_connection] Host: {host}", file=sys.stderr)

        connection = sql.connect(
            server_hostname=host,
            http_path="/sql/1.0/warehouses/5534359f9aac6560",
            access_token=user_token,
        )
        print("[get_databricks_connection] Connected successfully!", file=sys.stderr)
        return connection

    # Fallback: try service principal auth
    try:
        from databricks.sdk import WorkspaceClient
        print("[get_databricks_connection] No user token, trying SP auth...", file=sys.stderr)
        w = WorkspaceClient()
        auth_headers = w.config.authenticate()
        token = auth_headers.get("Authorization", "").replace("Bearer ", "")
        if token:
            host = w.config.host.replace("https://", "").replace("http://", "").strip("/")
            connection = sql.connect(
                server_hostname=host,
                http_path="/sql/1.0/warehouses/5534359f9aac6560",
                access_token=token,
            )
            print("[get_databricks_connection] Connected via SP auth!", file=sys.stderr)
            return connection
    except Exception as e:
        print(f"[get_databricks_connection] SP auth failed: {e}", file=sys.stderr)

    # Fallback: local dev with OAuth (only if secrets exist)
    secrets_path = os.path.expanduser("~/.streamlit/secrets.toml")
    if os.path.exists(secrets_path):
        host = st.secrets.get("DATABRICKS_HOST")
        http_path = st.secrets.get("DATABRICKS_HTTP_PATH")
        print(f"[get_databricks_connection] Using local OAuth with host: {host}", file=sys.stderr)
        return connect(
            server_hostname=host,
            http_path=http_path,
            auth_type="oauth"
        )

    raise Exception("No authentication available - no user token or secrets.toml found")

@st.cache_data
def load_trend_data(fy_start, fy_end, granularity="month"):
    """Query Databricks for time-series trends (weekly, monthly, or fiscal quarterly)"""
    try:
        conn = get_databricks_connection()
        cursor = conn.cursor()

        def period_expr(date_expr):
            """Bucket a date expression by granularity. Fiscal year starts Sept 1, so
            'quarter' buckets to fiscal quarters (Sep-Nov, Dec-Feb, Mar-May, Jun-Aug) by
            shifting 8 months before truncating to calendar quarter, then shifting back."""
            if granularity == 'quarter':
                return f"ADD_MONTHS(DATE_TRUNC('quarter', ADD_MONTHS({date_expr}, -8)), 8)"
            return f"DATE_TRUNC('{granularity}', {date_expr})"

        # Build the query with fiscal-aware period bucketing
        query = f"""
            WITH filtered_employees AS (
              SELECT DISTINCT e.EmployeeID, e.FullName_FNF, e.JobTitle, e.Department, e.Active
              FROM ea_prod.reference_gold.employee e
              INNER JOIN ea_prod.reference_gold.department_hierarchy dh
                ON e.DepartmentId = dh.DepartmentID AND dh.Current = true
              WHERE e.JobFamily = 'Client Consulting' AND e.Active = 1
                AND dh.DepartmentTypeName = 'Consulting - Generalist'
            ),
            dept_region AS (
              SELECT DISTINCT
                fe.Department,
                CASE
                  WHEN dh.Region = 'Americas' THEN 'Americas'
                  WHEN dh.Region = 'Europe' THEN 'EMEA'
                  WHEN dh.Region = 'Asia' THEN 'AsiaPac'
                  ELSE COALESCE(dh.Region, 'Unknown')
                END as Region
              FROM filtered_employees fe
              LEFT JOIN ea_prod.reference_gold.department_hierarchy dh
                ON fe.Department = dh.Department AND dh.Current = true
            ),
            case_trend AS (
              SELECT
                {period_expr('c.OpenedDate')} as Period,
                fe.Department,
                COUNT(*) as TotalCases,
                SUM(CAST(c.TimeToResolutionHrs AS DOUBLE)) as CaseHours
              FROM ea_prod.crmfacts_gold.cases c
              INNER JOIN filtered_employees fe ON c.CaseOwnerId = fe.EmployeeID
              WHERE c.OpenedDate >= '{fy_start}' AND c.OpenedDate <= '{fy_end}'
              GROUP BY {period_expr('c.OpenedDate')}, fe.Department
            ),
            meeting_trend AS (
              SELECT
                {period_expr('v.MeetingDate')} as Period,
                fe.Department,
                COUNT(DISTINCT v.MeetingId) as UniqueMeetings,
                ROUND(SUM(v.DurationInMinutes) / 60, 2) as MeetingHours
              FROM ea_prod.crmfacts_gold.vmeetings v
              INNER JOIN filtered_employees fe ON v.EmployeeName = fe.FullName_FNF
              WHERE v.MeetingDate >= '{fy_start}' AND v.MeetingDate <= '{fy_end}'
                AND v.DurationInMinutes <= 480
              GROUP BY {period_expr('v.MeetingDate')}, fe.Department
            ),
            call_email_trend AS (
              SELECT
                {period_expr('a.activity_date')} as Period,
                fe.Department,
                COUNT(DISTINCT CASE WHEN a.activity_type = 'Call' THEN a.id END) as CallCount,
                ROUND((((COUNT(DISTINCT CASE WHEN a.activity_type = 'Call' THEN a.id END) -
                         SUM(CASE WHEN a.activity_type = 'Call' AND a.connection_outcome_c = 'Connection Made' THEN 1 ELSE 0 END)) * 2) +
                        (SUM(CASE WHEN a.activity_type = 'Call' AND a.connection_outcome_c = 'Connection Made' THEN 1 ELSE 0 END) * 5)) / 60, 2) as CallHours,
                COUNT(DISTINCT CASE WHEN a.activity_type = 'Email' THEN a.id END) as EmailCount,
                ROUND((COUNT(DISTINCT CASE WHEN a.activity_type = 'Email' THEN a.id END) * 10) / 60, 2) as EmailHours
              FROM ea_prod.crmfacts_gold.vactivity a
              INNER JOIN filtered_employees fe ON a.activity_owner_id = fe.EmployeeID
              WHERE a.activity_date >= '{fy_start}' AND a.activity_date <= '{fy_end}'
                AND a.activity_type IN ('Call', 'Email')
              GROUP BY {period_expr('a.activity_date')}, fe.Department
            ),
            rpd_authored_trend AS (
              SELECT
                {period_expr('ra.`Created Date`')} as Period,
                fe.Department,
                COUNT(DISTINCT ra.`RPD#`) as RPDs_Authored
              FROM ea_prod.reporting_gold.pbi_rpd ra
              INNER JOIN filtered_employees fe ON TRY_CAST(ra.`Author ID` AS INT) = fe.EmployeeID
              WHERE ra.`Created Date` >= '{fy_start}' AND ra.`Created Date` <= '{fy_end}'
                AND TRY_CAST(ra.`Author ID` AS INT) IS NOT NULL
              GROUP BY {period_expr('ra.`Created Date`')}, fe.Department
            ),
            rpd_comments_trend AS (
              SELECT
                {period_expr("TO_DATE(CAST(rc.CommentDateKey AS STRING), 'yyyyMMdd')")} as Period,
                fe.Department,
                COUNT(*) as RPD_Comments
              FROM ea_prod.rpdfacts_gold.f_tr_rpd_comments rc
              INNER JOIN filtered_employees fe ON rc.CommenterId = fe.EmployeeID
              WHERE rc.CommentDateKey >= REPLACE('{fy_start}', '-', '')
                AND rc.CommentDateKey <= REPLACE('{fy_end}', '-', '')
              GROUP BY {period_expr("TO_DATE(CAST(rc.CommentDateKey AS STRING), 'yyyyMMdd')")}, fe.Department
            )
            SELECT
              COALESCE(c.Period, m.Period, e.Period, ra.Period, rc.Period) as Period,
              COALESCE(c.Department, m.Department, e.Department, ra.Department, rc.Department) as Department,
              dr.Region,
              COALESCE(c.TotalCases, 0) as TotalCases,
              COALESCE(c.CaseHours, 0) as CaseHours,
              COALESCE(m.UniqueMeetings, 0) as UniqueMeetings,
              COALESCE(m.MeetingHours, 0) as MeetingHours,
              COALESCE(e.CallCount, 0) as CallCount,
              COALESCE(e.CallHours, 0) as CallHours,
              COALESCE(e.EmailCount, 0) as EmailCount,
              COALESCE(e.EmailHours, 0) as EmailHours,
              COALESCE(ra.RPDs_Authored, 0) as RPDs_Authored,
              COALESCE(rc.RPD_Comments, 0) as RPD_Comments,
              ROUND((COALESCE(ra.RPDs_Authored, 0) * 20.0 + COALESCE(rc.RPD_Comments, 0) * 10.0) / 60.0, 2) as RPDHours,
              ROUND(
                COALESCE(c.CaseHours, 0) + COALESCE(m.MeetingHours, 0) +
                COALESCE(e.CallHours, 0) + COALESCE(e.EmailHours, 0) +
                (COALESCE(ra.RPDs_Authored, 0) * 20.0 + COALESCE(rc.RPD_Comments, 0) * 10.0) / 60.0, 2
              ) as TotalHours
            FROM case_trend c
            FULL OUTER JOIN meeting_trend m ON c.Period = m.Period AND c.Department = m.Department
            FULL OUTER JOIN call_email_trend e ON c.Period = e.Period AND c.Department = e.Department
            FULL OUTER JOIN rpd_authored_trend ra ON c.Period = ra.Period AND c.Department = ra.Department
            FULL OUTER JOIN rpd_comments_trend rc ON c.Period = rc.Period AND c.Department = rc.Department
            LEFT JOIN dept_region dr
              ON COALESCE(c.Department, m.Department, e.Department, ra.Department, rc.Department) = dr.Department
            ORDER BY Period, Department
        """

        cursor.execute(query)
        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()

        return df
    except Exception as e:
        st.error(f"Error loading trend data from Databricks: {e}")
        return None

@st.cache_data(ttl=86400)
def load_accounts_users_data():
    """Query Databricks for employee account/user coverage (cached 24 hours)"""
    import sys
    import time
    try:
        t0 = time.time()
        print(f"[load_accounts_users_data] Connecting...", file=sys.stderr)
        conn = get_databricks_connection()
        cursor = conn.cursor()

        query = """
            WITH filtered_employees AS (
              SELECT DISTINCT e.EmployeeID, e.FullName_FNF
              FROM ea_prod.reference_gold.employee e
              INNER JOIN ea_prod.reference_gold.department_hierarchy dh
                ON e.DepartmentId = dh.DepartmentID AND dh.Current = true
              WHERE e.JobFamily = 'Client Consulting' AND e.Active = 1
                AND dh.DepartmentTypeName = 'Consulting - Generalist'
            ),
            employee_locations AS (
              SELECT
                fe.EmployeeID,
                fe.FullName_FNF as EmployeeName,
                atm.LocationId
              FROM filtered_employees fe
              LEFT JOIN ea_prod.reference_gold.account_team_member atm
                ON fe.EmployeeID = atm.EmployeeId
            ),
            locations_with_users AS (
              SELECT
                el.EmployeeID,
                el.EmployeeName,
                u.mainlocationid,
                u.userid
              FROM employee_locations el
              LEFT JOIN ea_prod.reference_gold.user u
                ON el.LocationId = u.locationid
                AND u.current = 1
                AND u.active = 1
                AND u.usertype = 'Normal'
            )
            SELECT
              EmployeeID,
              EmployeeName,
              COUNT(DISTINCT mainlocationid) as AccountsCovered,
              COUNT(DISTINCT userid) as UsersCovered
            FROM locations_with_users
            GROUP BY EmployeeID, EmployeeName
        """

        print(f"[load_accounts_users_data] Executing query... (t+{time.time()-t0:.1f}s)", file=sys.stderr)
        cursor.execute(query)
        print(f"[load_accounts_users_data] Query executed, fetching... (t+{time.time()-t0:.1f}s)", file=sys.stderr)
        df = cursor.fetchall_arrow().to_pandas()
        print(f"[load_accounts_users_data] Done: {len(df)} rows (t+{time.time()-t0:.1f}s)", file=sys.stderr)
        cursor.close()
        return df
    except Exception as e:
        print(f"[load_accounts_users_data] Error: {e}", file=sys.stderr)
        return None

@st.cache_data
def load_data(fy_start, fy_end):
    """Query Databricks for combined metrics"""
    import sys
    try:
        print(f"[load_data] Connecting to Databricks...", file=sys.stderr)
        conn = get_databricks_connection()
        print(f"[load_data] Connected! Getting cursor...", file=sys.stderr)
        cursor = conn.cursor()
        print(f"[load_data] Cursor acquired, executing query...", file=sys.stderr)

        # Calculate business days and weeks (from notebook)
        from pandas.tseries.holiday import USFederalHolidayCalendar
        _holidays = USFederalHolidayCalendar().holidays(start=fy_start, end=fy_end).strftime("%Y-%m-%d").tolist()
        _end_exclusive = (_date.fromisoformat(fy_end) + timedelta(days=1)).isoformat()
        _biz_days = int(np.busday_count(fy_start, _end_exclusive, holidays=_holidays))
        _weeks = round(_biz_days / 5, 1)

        # Execute the combined metrics query from the notebook
        query = f"""
            WITH case_metrics AS (
              SELECT
                fe.Department,
                c.CaseOwner,
                COUNT(*) as TotalCases,
                SUM(CAST(c.TimeToResolutionHrs AS DOUBLE)) as TotalCaseTime
              FROM ea_prod.crmfacts_gold.cases c
              INNER JOIN (
                SELECT DISTINCT e.EmployeeID, e.FullName_FNF, e.JobTitle, e.Department
                FROM ea_prod.reference_gold.employee e
                INNER JOIN ea_prod.reference_gold.department_hierarchy dh
                  ON e.DepartmentId = dh.DepartmentID AND dh.Current = true
                WHERE e.JobFamily = 'Client Consulting' AND e.Active = 1
                  AND dh.DepartmentTypeName = 'Consulting - Generalist'
              ) fe ON c.CaseOwnerId = fe.EmployeeID
              WHERE c.OpenedDate >= '{fy_start}' AND c.OpenedDate <= '{fy_end}'
              GROUP BY fe.Department, c.CaseOwner
            ),
            filtered_employees AS (
              SELECT DISTINCT e.EmployeeID, e.FullName_FNF, e.JobTitle, e.Department, e.Active
              FROM ea_prod.reference_gold.employee e
              INNER JOIN ea_prod.reference_gold.department_hierarchy dh
                ON e.DepartmentId = dh.DepartmentID AND dh.Current = true
              WHERE e.JobFamily = 'Client Consulting' AND e.Active = 1
                AND dh.DepartmentTypeName = 'Consulting - Generalist'
            ),
            team_members AS (
              SELECT DISTINCT c.CaseOwner, c.CaseOwnerId, fe.Department
              FROM ea_prod.crmfacts_gold.cases c
              INNER JOIN filtered_employees fe ON c.CaseOwnerId = fe.EmployeeID
              WHERE c.OpenedDate >= '{fy_start}' AND c.OpenedDate <= '{fy_end}'
            ),
            delegation_metrics AS (
              SELECT
                c.CreatedBy as EmployeeName,
                tm.Department,
                COUNT(*) as TotalCasesCreated,
                SUM(CASE WHEN c.CreatedBy != c.CaseOwner THEN 1 ELSE 0 END) as CasesDelegatedAway
              FROM ea_prod.crmfacts_gold.cases c
              INNER JOIN team_members tm ON c.CreatedBy = tm.CaseOwner
              WHERE c.OpenedDate >= '{fy_start}' AND c.OpenedDate <= '{fy_end}'
              GROUP BY c.CreatedBy, tm.Department
            ),
            received_metrics AS (
              SELECT
                c.CaseOwner as EmployeeName,
                fe.Department,
                COUNT(*) as CurrentCasesOwned
              FROM ea_prod.crmfacts_gold.cases c
              INNER JOIN filtered_employees fe ON c.CaseOwnerId = fe.EmployeeID
              WHERE c.OpenedDate >= '{fy_start}' AND c.OpenedDate <= '{fy_end}'
              GROUP BY c.CaseOwner, fe.Department
            ),
            delegation_combined AS (
              SELECT
                COALESCE(dm.EmployeeName, rm.EmployeeName) as EmployeeName,
                COALESCE(dm.Department, rm.Department) as Department,
                COALESCE(dm.TotalCasesCreated, 0) as CasesCreated,
                COALESCE(dm.CasesDelegatedAway, 0) as CasesDelegated,
                (COALESCE(rm.CurrentCasesOwned, 0) - (COALESCE(dm.TotalCasesCreated, 0) - COALESCE(dm.CasesDelegatedAway, 0))) as CasesReceived
              FROM delegation_metrics dm
              FULL OUTER JOIN received_metrics rm ON dm.EmployeeName = rm.EmployeeName
            ),
            valid_meetings AS (
              SELECT MeetingId, EmployeeName
              FROM ea_prod.crmfacts_gold.vmeetings
              WHERE MeetingDate >= '{fy_start}' AND MeetingDate <= '{fy_end}'
                AND DurationInMinutes <= 480
              GROUP BY MeetingId, EmployeeName
              HAVING COUNT(DISTINCT IndividualId) > 0
                  OR MAX(CASE WHEN Description IS NOT NULL AND TRIM(Description) != '' THEN 1 ELSE 0 END) = 1
            ),
            meeting_dedup AS (
              SELECT
                v.MeetingId,
                v.EmployeeName,
                v.DurationInMinutes,
                ROW_NUMBER() OVER (PARTITION BY v.MeetingId, v.EmployeeName ORDER BY v.MeetingId) as rn
              FROM ea_prod.crmfacts_gold.vmeetings v
              INNER JOIN valid_meetings vm ON v.MeetingId = vm.MeetingId AND v.EmployeeName = vm.EmployeeName
              WHERE v.MeetingDate >= '{fy_start}' AND v.MeetingDate <= '{fy_end}'
                AND v.DurationInMinutes <= 480
            ),
            unique_meetings AS (
              SELECT MeetingId, EmployeeName, DurationInMinutes
              FROM meeting_dedup
              WHERE rn = 1
            ),
            attendee_dedup AS (
              SELECT
                v.MeetingId,
                v.EmployeeName,
                v.IndividualId,
                ROW_NUMBER() OVER (PARTITION BY v.MeetingId, v.EmployeeName, v.IndividualId ORDER BY v.MeetingId) as rn
              FROM ea_prod.crmfacts_gold.vmeetings v
              INNER JOIN valid_meetings vm ON v.MeetingId = vm.MeetingId AND v.EmployeeName = vm.EmployeeName
              WHERE v.MeetingDate >= '{fy_start}' AND v.MeetingDate <= '{fy_end}'
                AND v.DurationInMinutes <= 480
                AND v.IndividualId IS NOT NULL
            ),
            unique_attendees AS (
              SELECT MeetingId, EmployeeName, IndividualId
              FROM attendee_dedup
              WHERE rn = 1
            ),
            meeting_counts AS (
              SELECT
                fe.Department,
                um.EmployeeName,
                COUNT(DISTINCT um.MeetingId) as UniqueMeetings,
                ROUND(SUM(um.DurationInMinutes) / 60, 2) as MeetingHours
              FROM unique_meetings um
              INNER JOIN filtered_employees fe ON um.EmployeeName = fe.FullName_FNF
              GROUP BY fe.Department, um.EmployeeName
            ),
            attendee_counts AS (
              SELECT
                ua.EmployeeName,
                COUNT(ua.IndividualId) as TotalAttendeeSlots,
                COUNT(DISTINCT ua.IndividualId) as UniqueClientsMetWith
              FROM unique_attendees ua
              GROUP BY ua.EmployeeName
            ),
            meeting_metrics AS (
              SELECT
                mc.Department,
                mc.EmployeeName,
                mc.UniqueMeetings,
                COALESCE(ac.UniqueClientsMetWith, 0) as UniqueClientsMetWith,
                ROUND(COALESCE(ac.TotalAttendeeSlots, 0) * 1.0 / NULLIF(mc.UniqueMeetings, 0), 2) as UsersPerMeeting,
                mc.MeetingHours
              FROM meeting_counts mc
              LEFT JOIN attendee_counts ac ON mc.EmployeeName = ac.EmployeeName
            ),
            deduplicated_activities AS (
              SELECT
                id as ActivityId,
                activity_owner_id as EmployeeId,
                activity_type as ActivityType,
                connection_outcome_c as ConnectionOutcome,
                ROW_NUMBER() OVER (PARTITION BY id, activity_owner_id ORDER BY id) as rn
              FROM ea_prod.crmfacts_gold.vactivity
              WHERE activity_date >= '{fy_start}' AND activity_date <= '{fy_end}'
                AND activity_type IN ('Call', 'Email')
            ),
            unique_activities AS (
              SELECT ActivityId, EmployeeId, ActivityType, ConnectionOutcome
              FROM deduplicated_activities
              WHERE rn = 1
            ),
            case_owners AS (
              SELECT EmployeeID as CaseOwnerId, FullName_FNF as CaseOwner, Department
              FROM filtered_employees
            ),
            call_email_metrics AS (
              SELECT
                co.Department,
                co.CaseOwner as EmployeeName,
                COUNT(DISTINCT CASE WHEN a.ActivityType = 'Call' THEN a.ActivityId END) as UniqueCalls,
                SUM(CASE WHEN a.ActivityType = 'Call' AND a.ConnectionOutcome = 'Connection Made' THEN 1 ELSE 0 END) as ConnectionsMade,
                ROUND((SUM(CASE WHEN a.ActivityType = 'Call' AND a.ConnectionOutcome = 'Connection Made' THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT CASE WHEN a.ActivityType = 'Call' THEN a.ActivityId END), 0)) * 100, 2) as HitRatePct,
                ROUND((((COUNT(DISTINCT CASE WHEN a.ActivityType = 'Call' THEN a.ActivityId END) - SUM(CASE WHEN a.ActivityType = 'Call' AND a.ConnectionOutcome = 'Connection Made' THEN 1 ELSE 0 END)) * 2) + (SUM(CASE WHEN a.ActivityType = 'Call' AND a.ConnectionOutcome = 'Connection Made' THEN 1 ELSE 0 END) * 5)) / 60, 2) as CallTimeSpentHours,
                COUNT(DISTINCT CASE WHEN a.ActivityType = 'Email' THEN a.ActivityId END) as EmailCount,
                ROUND((COUNT(DISTINCT CASE WHEN a.ActivityType = 'Email' THEN a.ActivityId END) * 10) / 60, 2) as EmailHours
              FROM unique_activities a
              INNER JOIN case_owners co ON a.EmployeeId = co.CaseOwnerId
              GROUP BY co.Department, co.CaseOwner
            ),
            rpd_comments_agg AS (
              SELECT
                CommenterId as EmployeeId,
                COUNT(*) as RPD_Comments,
                COUNT(DISTINCT MessageKey) as RPDs_CommentedOn
              FROM ea_prod.rpdfacts_gold.f_tr_rpd_comments
              WHERE CommentDateKey >= REPLACE('{fy_start}', '-', '')
                AND CommentDateKey <= REPLACE('{fy_end}', '-', '')
              GROUP BY CommenterId
            ),
            rpd_authored_agg AS (
              SELECT
                TRY_CAST(`Author ID` AS INT) as AuthorId,
                COUNT(DISTINCT `RPD#`) as RPDs_Authored
              FROM ea_prod.reporting_gold.pbi_rpd
              WHERE `Created Date` >= '{fy_start}' AND `Created Date` <= '{fy_end}'
                AND TRY_CAST(`Author ID` AS INT) IS NOT NULL
              GROUP BY TRY_CAST(`Author ID` AS INT)
            )
            SELECT
              CASE
                WHEN dh_region.Region = 'Americas' THEN 'Americas'
                WHEN dh_region.Region = 'Europe' THEN 'EMEA'
                WHEN dh_region.Region = 'Asia' THEN 'AsiaPac'
                ELSE COALESCE(dh_region.Region, 'Unknown')
              END as Region,
              fe.EmployeeID,
              fe.Department,
              fe.FullName_FNF as EmployeeName,
              REPLACE(fe.JobTitle, ', Client Solutions', '') as JobTitle,
              ROUND(DATEDIFF(CURRENT_DATE(), hd.HireDate) / 365.25, 1) as TenureYears,
              COALESCE(d.CasesCreated, 0) as CasesCreated,
              COALESCE(d.CasesDelegated, 0) as CasesDelegated,
              COALESCE(d.CasesReceived, 0) as CasesReceived,
              COALESCE(c.TotalCases, 0) as TotalCases,
              ROUND(COALESCE(c.TotalCaseTime, 0), 2) as TotalCaseHours,
              COALESCE(m.UniqueMeetings, 0) as UniqueMeetings,
              COALESCE(m.UniqueClientsMetWith, 0) as UniqueClientsMetWith,
              COALESCE(m.UsersPerMeeting, 0) as UsersPerMeeting,
              COALESCE(m.MeetingHours, 0) as MeetingHours,
              COALESCE(ce.UniqueCalls, 0) as UniqueCalls,
              COALESCE(ce.ConnectionsMade, 0) as ConnectionsMade,
              COALESCE(ce.HitRatePct, 0) as HitRatePct,
              COALESCE(ce.CallTimeSpentHours, 0) as CallTimeSpentHours,
              COALESCE(ce.EmailCount, 0) as EmailCount,
              COALESCE(ce.EmailHours, 0) as EmailHours,
              COALESCE(ra.RPDs_Authored, 0) as RPDs_Authored,
              COALESCE(rpd.RPD_Comments, 0) as RPD_Comments,
              ROUND((COALESCE(ra.RPDs_Authored, 0) * 20.0 + COALESCE(rpd.RPD_Comments, 0) * 10.0) / 60.0, 1) as RPD_EstHours,
              ROUND(
                COALESCE(c.TotalCaseTime, 0) +
                COALESCE(m.MeetingHours, 0) +
                COALESCE(ce.CallTimeSpentHours, 0) +
                COALESCE(ce.EmailHours, 0) +
                (COALESCE(ra.RPDs_Authored, 0) * 20.0 + COALESCE(rpd.RPD_Comments, 0) * 10.0) / 60.0, 2
              ) as TotalTimeSpentHours,
              ROUND(
                (
                  COALESCE(c.TotalCaseTime, 0) +
                  COALESCE(m.MeetingHours, 0) +
                  COALESCE(ce.CallTimeSpentHours, 0) +
                  COALESCE(ce.EmailHours, 0) +
                  (COALESCE(ra.RPDs_Authored, 0) * 20.0 + COALESCE(rpd.RPD_Comments, 0) * 10.0) / 60.0
                ) /
                NULLIF({_weeks}, 0), 2
              ) as HoursSpentPerWeek,
              ROUND(COALESCE(c.TotalCaseTime, 0) / NULLIF(
                COALESCE(c.TotalCaseTime, 0) + COALESCE(m.MeetingHours, 0) +
                COALESCE(ce.CallTimeSpentHours, 0) + COALESCE(ce.EmailHours, 0) +
                (COALESCE(ra.RPDs_Authored, 0) * 20.0 + COALESCE(rpd.RPD_Comments, 0) * 10.0) / 60.0, 0) * 100, 1) as CasesPct,
              ROUND(COALESCE(m.MeetingHours, 0) / NULLIF(
                COALESCE(c.TotalCaseTime, 0) + COALESCE(m.MeetingHours, 0) +
                COALESCE(ce.CallTimeSpentHours, 0) + COALESCE(ce.EmailHours, 0) +
                (COALESCE(ra.RPDs_Authored, 0) * 20.0 + COALESCE(rpd.RPD_Comments, 0) * 10.0) / 60.0, 0) * 100, 1) as MeetingsPct,
              ROUND(COALESCE(ce.CallTimeSpentHours, 0) / NULLIF(
                COALESCE(c.TotalCaseTime, 0) + COALESCE(m.MeetingHours, 0) +
                COALESCE(ce.CallTimeSpentHours, 0) + COALESCE(ce.EmailHours, 0) +
                (COALESCE(ra.RPDs_Authored, 0) * 20.0 + COALESCE(rpd.RPD_Comments, 0) * 10.0) / 60.0, 0) * 100, 1) as CallsPct,
              ROUND(COALESCE(ce.EmailHours, 0) / NULLIF(
                COALESCE(c.TotalCaseTime, 0) + COALESCE(m.MeetingHours, 0) +
                COALESCE(ce.CallTimeSpentHours, 0) + COALESCE(ce.EmailHours, 0) +
                (COALESCE(ra.RPDs_Authored, 0) * 20.0 + COALESCE(rpd.RPD_Comments, 0) * 10.0) / 60.0, 0) * 100, 1) as EmailsPct,
              ROUND((COALESCE(ra.RPDs_Authored, 0) * 20.0 + COALESCE(rpd.RPD_Comments, 0) * 10.0) / 60.0 / NULLIF(
                COALESCE(c.TotalCaseTime, 0) + COALESCE(m.MeetingHours, 0) +
                COALESCE(ce.CallTimeSpentHours, 0) + COALESCE(ce.EmailHours, 0) +
                (COALESCE(ra.RPDs_Authored, 0) * 20.0 + COALESCE(rpd.RPD_Comments, 0) * 10.0) / 60.0, 0) * 100, 1) as RPDsPct
            FROM filtered_employees fe
            LEFT JOIN ea_prod.reference_gold.department_hierarchy dh_region
              ON fe.Department = dh_region.Department AND dh_region.Current = true
            LEFT JOIN ea_dev.reporting_gold.sl_employeedepartment_history hd
              ON fe.EmployeeID = hd.EmployeeId AND hd.Current = true
            LEFT JOIN case_metrics c        ON fe.FullName_FNF = c.CaseOwner
            LEFT JOIN delegation_combined d ON fe.FullName_FNF = d.EmployeeName
            LEFT JOIN meeting_metrics m     ON fe.FullName_FNF = m.EmployeeName
            LEFT JOIN call_email_metrics ce ON fe.FullName_FNF = ce.EmployeeName
            LEFT JOIN rpd_comments_agg rpd ON fe.EmployeeID = rpd.EmployeeId
            LEFT JOIN rpd_authored_agg ra  ON fe.EmployeeID = ra.AuthorId
            ORDER BY Region, fe.Department, COALESCE(c.TotalCases, 0) DESC
        """

        print(f"[load_data] Executing query...", file=sys.stderr)
        cursor.execute(query)
        print(f"[load_data] Query executed, fetching results...", file=sys.stderr)
        df = cursor.fetchall_arrow().to_pandas()
        print(f"[load_data] Results fetched: {len(df)} rows", file=sys.stderr)
        cursor.close()

        return df
    except Exception as e:
        import traceback
        error_msg = f"Error loading data: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        st.error(error_msg)
        return None

# ============================================================================
# SIDEBAR: FILTERS & DATE RANGE & CAPACITY TARGET
# ============================================================================

st.sidebar.header("Filters & Settings")

# Date range picker - dynamic fiscal year (9/1 to 8/31)
today = datetime.today()
current_year = today.year
current_month = today.month

if current_month >= 9:
    default_start = datetime(current_year, 9, 1)
else:
    default_start = datetime(current_year - 1, 9, 1)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=default_start)
with col2:
    end_date = st.date_input("End Date", value=today)

# Convert to ISO format for the query
fy_start = start_date.isoformat()
fy_end = end_date.isoformat()

# Load data with selected dates
df = load_data(fy_start, fy_end)

if df is None:
    st.stop()

# Load accounts/users coverage separately (cached 24h) and merge
accounts_users_df = load_accounts_users_data()
if accounts_users_df is not None and len(accounts_users_df) > 0:
    df = df.merge(
        accounts_users_df[['EmployeeID', 'AccountsCovered', 'UsersCovered']],
        on='EmployeeID',
        how='left'
    )
    df['AccountsCovered'] = df['AccountsCovered'].fillna(0).astype(int)
    df['UsersCovered'] = df['UsersCovered'].fillna(0).astype(int)
else:
    df['AccountsCovered'] = 0
    df['UsersCovered'] = 0

# Get unique regions and departments
regions = sorted(df['Region'].unique().tolist())
all_departments = sorted(df['Department'].unique().tolist())

# Region filter
st.sidebar.markdown("**Region**")
selected_regions = st.sidebar.multiselect(
    "Select region(s)",
    options=regions,
    default=[],
    label_visibility="collapsed"
)

# Get departments in selected regions (or all if no region selected)
if selected_regions:
    depts_in_regions = sorted(df[df['Region'].isin(selected_regions)]['Department'].unique().tolist())
    default_depts = depts_in_regions
else:
    depts_in_regions = all_departments
    default_depts = []

# Department filter
st.sidebar.markdown("**Department(s)**")
selected_dept = st.sidebar.multiselect(
    "Select department(s)",
    options=depts_in_regions,
    default=default_depts,
    label_visibility="collapsed"
)

# Reset button
if st.sidebar.button("🔄 Reset Filters", use_container_width=True):
    st.rerun()

# Filter by selected regions and departments
if selected_regions or selected_dept:
    filtered_df = df.copy()
    if selected_regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_regions)]
    if selected_dept:
        filtered_df = filtered_df[filtered_df['Department'].isin(selected_dept)]
else:
    filtered_df = df.copy()

# Calculate group average for capacity comparison
group_avg_hours = filtered_df['HoursSpentPerWeek'].mean() if len(filtered_df) > 0 else 0

# Capacity variance slider
st.sidebar.markdown("**Capacity Variance**")
variance_pct = st.sidebar.slider("Allowed variance from avg (%)", min_value=0, max_value=50, value=0, step=1)

# Add capacity status columns (based on group average and variance)
filtered_df['CapacityStatus'], filtered_df['CapacityColor'], filtered_df['CapacityIcon'] = zip(
    *filtered_df['HoursSpentPerWeek'].apply(lambda x: capacity_status(x, group_avg_hours, variance_pct))
)

# ============================================================================
# CALCULATE KEY METRICS
# ============================================================================

total_hours = filtered_df['TotalTimeSpentHours'].sum() if len(filtered_df) > 0 else 0
avg_hours_per_week = filtered_df['HoursSpentPerWeek'].mean() if len(filtered_df) > 0 else 0
total_employees = len(filtered_df)
total_departments = filtered_df['Department'].nunique() if len(filtered_df) > 0 else 0
total_cases = filtered_df['TotalCases'].sum() if len(filtered_df) > 0 else 0
total_meetings = filtered_df['UniqueMeetings'].sum() if len(filtered_df) > 0 else 0

# Capacity counts
capacity_counts = filtered_df['CapacityStatus'].value_counts()
on_target_count = capacity_counts.get('On Target', 0)
under_count = capacity_counts.get('Under', 0)
over_count = capacity_counts.get('Over', 0)

# ============================================================================
# TAB STRUCTURE
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Team", "Individual", "Trends", "Data & Export", "Methodology"])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    st.subheader("Executive Summary")

    # KPI row with cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        with st.container(border=True):
            st.metric("Total Hours", f"{total_hours:,.0f}h")

    with col2:
        with st.container(border=True):
            st.metric("Avg Hrs/Week", f"{avg_hours_per_week:.1f}h")

    with col3:
        with st.container(border=True):
            st.metric("Active Employees", f"{total_employees}")

    with col4:
        with st.container(border=True):
            st.metric("Departments", f"{total_departments}")

    with col5:
        with st.container(border=True):
            st.metric("Total Cases", f"{total_cases:,.0f}")

    with col6:
        with st.container(border=True):
            st.metric("Total Meetings", f"{total_meetings:,.0f}")

    st.caption("ℹ️ Total Hours and Avg Hrs/Week include estimated time for Calls, Emails, and RPDs — see the Methodology tab for how each is calculated.")

    st.divider()

    # Capacity summary
    if variance_pct > 0:
        tolerance = group_avg_hours * (variance_pct / 100)
        st.write(f"**Group Average:** {group_avg_hours:.1f} hrs/week | **On-Target Range:** {group_avg_hours - tolerance:.1f} — {group_avg_hours + tolerance:.1f} hrs/week (±{variance_pct}%)")
    else:
        st.write(f"**Group Average:** {group_avg_hours:.1f} hrs/week | Variance: 0% (exact match only)")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.metric("On Target", on_target_count)

    with col2:
        with st.container(border=True):
            st.metric("Under Capacity", under_count)

    with col3:
        with st.container(border=True):
            st.metric("Over Capacity", over_count)

    st.divider()

    # Client Coverage Section
    st.subheader("Client Coverage")

    total_accounts_covered = filtered_df['AccountsCovered'].sum() if len(filtered_df) > 0 else 0
    total_users_covered = filtered_df['UsersCovered'].sum() if len(filtered_df) > 0 else 0
    avg_accounts_per_employee = filtered_df['AccountsCovered'].mean() if len(filtered_df) > 0 else 0
    avg_users_per_employee = filtered_df['UsersCovered'].mean() if len(filtered_df) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.metric("Total Accounts Covered", f"{total_accounts_covered:,.0f}")
    with col2:
        with st.container(border=True):
            st.metric("Total Users Covered", f"{total_users_covered:,.0f}")
    with col3:
        with st.container(border=True):
            st.metric("Avg Accounts / Employee", f"{avg_accounts_per_employee:,.1f}")
    with col4:
        with st.container(border=True):
            st.metric("Avg Users / Employee", f"{avg_users_per_employee:,.1f}")

    if len(filtered_df) > 0:
        coverage_by_dept = filtered_df.groupby('Department')[['AccountsCovered', 'UsersCovered']].sum().reset_index()
        coverage_by_dept = coverage_by_dept.sort_values('AccountsCovered', ascending=True)

        fig_coverage = go.Figure()
        fig_coverage.add_trace(go.Bar(
            y=coverage_by_dept['Department'],
            x=coverage_by_dept['AccountsCovered'],
            orientation='h',
            name='Accounts Covered',
            marker=dict(color=CATEGORICAL[0]),
            hovertemplate='<b>%{y}</b><br>Accounts: %{x:,.0f}<extra></extra>'
        ))
        fig_coverage.add_trace(go.Bar(
            y=coverage_by_dept['Department'],
            x=coverage_by_dept['UsersCovered'],
            orientation='h',
            name='Users Covered',
            marker=dict(color=CATEGORICAL[1]),
            hovertemplate='<b>%{y}</b><br>Users: %{x:,.0f}<extra></extra>'
        ))
        fig_coverage.update_layout(
            height=max(300, 40 * len(coverage_by_dept)),
            barmode='group',
            xaxis_title='Count',
            yaxis_title='',
            legend=dict(yanchor="bottom", y=1.02, xanchor="left", x=0, orientation="h"),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_coverage, use_container_width=True)

    st.divider()

    # Time Allocation Section (enhanced)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Workforce Activity Distribution")
    with col2:
        activity_drill_dept = st.selectbox(
            "Drill into:",
            options=["All Departments"] + sorted(filtered_df['Department'].unique().tolist()),
            key="activity_drill",
            label_visibility="collapsed"
        )
    st.caption("ℹ️ Cases & Meetings hours are actual tracked time. Calls, Emails, and RPDs hours are estimates based on fixed time-per-activity assumptions — see the Methodology tab for details.")

    # Filter data for activity view
    if activity_drill_dept == "All Departments":
        activity_view_df = filtered_df
        view_label = "Org-Wide"
    else:
        activity_view_df = filtered_df[filtered_df['Department'] == activity_drill_dept]
        view_label = activity_drill_dept

    activity_data = {
        'Cases': activity_view_df['TotalCaseHours'].sum(),
        'Meetings': activity_view_df['MeetingHours'].sum(),
        'Calls': activity_view_df['CallTimeSpentHours'].sum(),
        'Emails': activity_view_df['EmailHours'].sum(),
        'RPDs': activity_view_df['RPD_EstHours'].sum()
    }

    # Charts row (only show activities with > 0 hours)
    activity_data_for_charts = {k: v for k, v in activity_data.items() if v > 0}

    col1, col2 = st.columns(2)

    with col1:
        if activity_data_for_charts:
            colors_list = [ACTIVITY_COLORS.get(activity, '#999999') for activity in activity_data_for_charts.keys()]
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(activity_data_for_charts.keys()),
                values=list(activity_data_for_charts.values()),
                marker=dict(colors=colors_list),
                hovertemplate='<b>%{label}</b><br>Hours: %{value:,.0f}<br>%{percent}<extra></extra>'
            )])
            fig_pie.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No activity hours recorded")

    with col2:
        if activity_data_for_charts:
            fig_bar = go.Figure(data=[go.Bar(
                y=list(activity_data_for_charts.keys()),
                x=list(activity_data_for_charts.values()),
                orientation='h',
                marker=dict(color=[ACTIVITY_COLORS.get(activity, '#999999') for activity in activity_data_for_charts.keys()]),
                hovertemplate='<b>%{y}</b><br>Hours: %{x:,.0f}<extra></extra>',
                text=[f"{v:,.0f}h" for v in activity_data_for_charts.values()],
                textposition='auto'
            )])
            fig_bar.update_layout(
                height=400,
                showlegend=False,
                yaxis_title="",
                xaxis_title="Hours",
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Summary metrics row (all 5 activities, including zeros)
    activity_df = pd.DataFrame({
        'Activity': list(activity_data.keys()),
        'Hours': [float(v) for v in activity_data.values()]
    })
    total_hours = activity_df['Hours'].sum()
    activity_df['% of Total'] = (activity_df['Hours'] / total_hours * 100).round(1) if total_hours > 0 else 0
    activity_df = activity_df.sort_values('Hours', ascending=False)

    activity_proxy_notes = {
        'Cases': "Actual tracked time (TimeToResolutionHrs)",
        'Meetings': "Actual tracked time (meeting duration)",
        'Calls': "Estimate: 2 min/missed call + 5 min/connected call",
        'Emails': "Estimate: 10 min per email",
        'RPDs': "Estimate: 20 min/authored + 10 min/comment",
    }

    metric_cols = st.columns(len(activity_df))
    for idx, (col, (_, row)) in enumerate(zip(metric_cols, activity_df.iterrows())):
        with col:
            with st.container(border=True):
                st.metric(
                    row['Activity'],
                    f"{row['Hours']:,.0f}h",
                    help=activity_proxy_notes.get(row['Activity'], '')
                )
                st.caption(f"{row['% of Total']:.0f}% of total")

    st.divider()

    # Detailed employee table by department (with department averages embedded)
    st.subheader("Employee Metrics by Department")
    st.caption("ℹ️ HoursSpentPerWeek, CallsPct, EmailsPct, and RPDsPct include estimated time (Calls/Emails/RPDs are not directly tracked) — see the Methodology tab for how each is calculated.")

    for dept in sorted(filtered_df['Department'].unique()):
        dept_employees = filtered_df[filtered_df['Department'] == dept].sort_values('HoursSpentPerWeek', ascending=False)

        display_cols = ['EmployeeName', 'JobTitle', 'TenureYears', 'AccountsCovered', 'UsersCovered',
                       'HoursSpentPerWeek',
                       'TotalCases', 'UniqueMeetings', 'UniqueCalls', 'HitRatePct', 'EmailCount', 'RPDs_Authored',
                       'CasesPct', 'MeetingsPct', 'CallsPct', 'EmailsPct', 'RPDsPct']

        # Filter to only columns that exist
        display_cols = [col for col in display_cols if col in dept_employees.columns]

        with st.expander(f"📂 {dept} ({len(dept_employees)} employees)", expanded=False):
            # Create department average row (rendered as a pinned row in the grid)
            dept_avg_row = {}
            for col in display_cols:
                if col == 'EmployeeName':
                    dept_avg_row[col] = 'DEPT AVG'
                elif col == 'JobTitle':
                    dept_avg_row[col] = ''
                elif dept_employees[col].dtype.kind in 'if':
                    dept_avg_row[col] = float(dept_employees[col].mean())
                else:
                    dept_avg_row[col] = ''

            display_df = dept_employees[display_cols].copy()
            pct_cols = ['CasesPct', 'MeetingsPct', 'CallsPct', 'EmailsPct', 'RPDsPct', 'HitRatePct']

            # Pinned DEPT AVG row is gray/bold; employee rows are colored by variance vs. avg
            variance_cell_style = JsCode("""
                function(params) {
                    if (params.node.rowPinned) {
                        return {backgroundColor: '#f0f2f6', fontWeight: 'bold'};
                    }
                    var val = params.value;
                    var avg = params.colDef.avgValue;
                    if (val == null || avg == null || avg === 0 || val === 0) { return {}; }
                    var variance = ((val - avg) / avg) * 100;
                    if (variance >= 15) {
                        var intensity = Math.min(Math.abs(variance) / 100, 1);
                        return {backgroundColor: 'rgba(12, 163, 12, ' + (intensity * 0.7) + ')'};
                    } else if (variance <= -15) {
                        var intensity = Math.min(Math.abs(variance) / 100, 1);
                        return {backgroundColor: 'rgba(211, 59, 59, ' + (intensity * 0.7) + ')'};
                    } else {
                        return {backgroundColor: 'rgba(250, 178, 25, 0.5)'};
                    }
                }
            """)
            pct_formatter = JsCode("function(params) { return params.value != null ? params.value.toFixed(1) + '%' : ''; }")
            hours_formatter = JsCode("function(params) { return params.value != null ? params.value.toFixed(2) : ''; }")
            int_formatter = JsCode("function(params) { return params.value != null ? Math.trunc(params.value).toString() : ''; }")

            # Deterministic content-based width for the two text columns (no JS auto-size timing issues)
            def text_col_width(col_name, min_width=90, max_width=260):
                values = [dept_avg_row.get(col_name, '')] + display_df[col_name].astype(str).tolist()
                max_len = max([len(col_name)] + [len(str(v)) for v in values])
                return int(min(max_width, max(min_width, max_len * 8 + 24)))

            gb = GridOptionsBuilder.from_dataframe(display_df)
            gb.configure_default_column(resizable=True, sortable=True, filter=False, cellStyle=variance_cell_style)

            pinned_cols = {'EmployeeName'}

            for col in display_cols:
                pin_opts = {'pinned': 'left'} if col in pinned_cols else {}

                if col in ('EmployeeName', 'JobTitle'):
                    gb.configure_column(col, width=text_col_width(col), **pin_opts)
                elif col == 'TenureYears':
                    gb.configure_column(col, **pin_opts)  # no variance heatmap — display raw value
                elif col in ('AccountsCovered', 'UsersCovered'):
                    gb.configure_column(col, valueFormatter=int_formatter, **pin_opts)  # no variance heatmap
                elif col in pct_cols:
                    gb.configure_column(col, valueFormatter=pct_formatter, avgValue=dept_avg_row[col], **pin_opts)
                elif col == 'HoursSpentPerWeek':
                    gb.configure_column(col, valueFormatter=hours_formatter, avgValue=dept_avg_row[col], **pin_opts)
                else:
                    gb.configure_column(col, valueFormatter=int_formatter, avgValue=dept_avg_row[col], **pin_opts)

            grid_options = gb.build()
            grid_options['autoSizeStrategy'] = None
            grid_options['pinnedTopRowData'] = [dept_avg_row]

            AgGrid(
                display_df,
                gridOptions=grid_options,
                height=460,
                theme='streamlit',
                allow_unsafe_jscode=True,
                key=f"aggrid_{dept}"
            )

# ============================================================================
# TAB 2: TEAM
# ============================================================================

with tab2:
    st.subheader("Team Summary")

    team_summary = filtered_df.groupby('Department').agg({
        'TotalTimeSpentHours': 'sum',
        'HoursSpentPerWeek': 'mean',
        'TotalCases': 'sum',
        'UniqueMeetings': 'sum',
        'UniqueCalls': 'sum',
        'EmailCount': 'sum'
    }).round(2)

    team_summary = team_summary.sort_values('TotalTimeSpentHours', ascending=False)

    if len(team_summary) > 0:
        col1, col2 = st.columns(2)

        with col1:
            team_hrs_sorted = team_summary.reset_index().sort_values('TotalTimeSpentHours', ascending=True)
            fig_team_hrs = px.bar(
                team_hrs_sorted,
                x='TotalTimeSpentHours',
                y='Department',
                orientation='h',
                title='Total Hours by Department',
                labels={'TotalTimeSpentHours': 'Hours', 'Department': ''},
                color='HoursSpentPerWeek',
                color_continuous_scale=SEQUENTIAL_BLUE
            )
            fig_team_hrs.update_layout(height=max(400, 30 * len(team_hrs_sorted)), margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_team_hrs, use_container_width=True)

        with col2:
            fig_team_metrics = px.scatter(
                team_summary.reset_index(),
                x='HoursSpentPerWeek',
                y='TotalCases',
                size='TotalTimeSpentHours',
                hover_name='Department',
                title='Department Workload: Hours/Week vs Cases',
                labels={'HoursSpentPerWeek': 'Hours per Week', 'TotalCases': 'Cases Handled'}
            )
            fig_team_metrics.update_layout(height=400)
            st.plotly_chart(fig_team_metrics, use_container_width=True)

        st.divider()
        st.subheader("Department Activity Mix")
        st.caption("ℹ️ Cases & Meetings % are based on actual tracked time. Calls, Emails, and RPDs % are based on time estimates — see the Methodology tab.")

        # Heatmap: activity allocation % by department
        dept_activity_mix = filtered_df.groupby('Department')[['CasesPct', 'MeetingsPct', 'CallsPct', 'EmailsPct', 'RPDsPct']].mean()
        dept_activity_mix.columns = ['Cases', 'Meetings', 'Calls', 'Emails', 'RPDs']

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=dept_activity_mix.values,
            x=['Cases', 'Meetings', 'Calls', 'Emails', 'RPDs'],
            y=dept_activity_mix.index,
            colorscale=SEQUENTIAL_BLUE,
            hovertemplate='<b>%{y}</b> | %{x}<br>%{z:.1f}%<extra></extra>',
            text=np.round(dept_activity_mix.values, 1),
            texttemplate='%{text:.1f}%',
            textfont={"size": 10}
        ))
        fig_heatmap.update_layout(height=max(300, 30 * len(dept_activity_mix)), margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.divider()
        st.subheader("Team Summary Table")

        totals_row = pd.DataFrame({
            'TotalTimeSpentHours': [filtered_df['TotalTimeSpentHours'].sum().round(2)],
            'HoursSpentPerWeek': [filtered_df['HoursSpentPerWeek'].mean().round(2)],
            'TotalCases': [filtered_df['TotalCases'].sum()],
            'UniqueMeetings': [filtered_df['UniqueMeetings'].sum()],
            'UniqueCalls': [filtered_df['UniqueCalls'].sum()],
            'EmailCount': [filtered_df['EmailCount'].sum()]
        }, index=['TOTAL'])
        team_summary_display = pd.concat([team_summary, totals_row])

        st.dataframe(
            team_summary_display.style.format({
                'TotalTimeSpentHours': '{:.2f}',
                'HoursSpentPerWeek': '{:.2f}'
            }).apply(
                lambda row: ['font-weight: bold; background-color: #f0f2f6'] * len(row) if row.name == 'TOTAL' else [''] * len(row),
                axis=1
            ),
            use_container_width=True,
            height=(len(team_summary_display) + 1) * 35 + 3
        )

# ============================================================================
# TAB 3: INDIVIDUAL
# ============================================================================

with tab3:
    st.subheader("Individual Employee Details")
    st.info("🚧 Work in Progress")

    if len(filtered_df) > 0:
        filtered_df['emp_label'] = filtered_df['EmployeeName'] + ' (' + filtered_df['TenureYears'].astype(str) + ' yrs)'

        col1, col2 = st.columns(2)
        with col1:
            indiv_dept_options = ["All Departments"] + sorted(filtered_df['Department'].unique().tolist())
            indiv_dept = st.selectbox("Department", indiv_dept_options, key="indiv_dept_filter")

        indiv_scope_df = filtered_df if indiv_dept == "All Departments" else filtered_df[filtered_df['Department'] == indiv_dept]
        emp_labels = sorted(indiv_scope_df['emp_label'].tolist())

        with col2:
            if len(emp_labels) > 0:
                selected_employee = st.selectbox("Select Employee", emp_labels, key="indiv_emp_filter")

        if len(emp_labels) > 0:
            selected_row = filtered_df[filtered_df['emp_label'] == selected_employee]
            if len(selected_row) > 0:
                emp_data = selected_row.iloc[0]
                # Databricks can return numeric fields as decimal.Decimal, which doesn't mix with
                # float in arithmetic (raises TypeError) — normalize once for all downstream math.
                for _col in emp_data.index:
                    if isinstance(emp_data[_col], Decimal):
                        emp_data[_col] = float(emp_data[_col])

                # Employee info with capacity badge
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    with st.container(border=True):
                        st.metric("Total Hours", f"{emp_data['TotalTimeSpentHours']:.1f}h")
                with col2:
                    with st.container(border=True):
                        st.metric(
                            "Hours/Week",
                            f"{emp_data['HoursSpentPerWeek']:.1f}h",
                            help=f"{emp_data['HoursSpentPerWeek'] - group_avg_hours:+.1f}h vs {group_avg_hours:.1f}h group average"
                        )
                with col3:
                    with st.container(border=True):
                        st.metric("Title", emp_data.get('JobTitle', 'N/A'))
                with col4:
                    with st.container(border=True):
                        st.metric("Tenure (Yrs)", f"{emp_data.get('TenureYears', 'N/A')}")
                with col5:
                    with st.container(border=True):
                        capacity_emoji = {"Under": "🟡", "On Target": "🟢", "Over": "🔴"}.get(emp_data['CapacityStatus'], "")
                        st.metric("Capacity", f"{capacity_emoji} {emp_data['CapacityStatus']}")

                st.caption("ℹ️ Total Hours and Hours/Week include estimated time for Calls, Emails, and RPDs — see the Methodology tab for how each is calculated.")

                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.metric("Accounts Covered", f"{emp_data.get('AccountsCovered', 0):,.0f}")
                with col2:
                    with st.container(border=True):
                        st.metric("Users Covered", f"{emp_data.get('UsersCovered', 0):,.0f}")

                if variance_pct > 0:
                    tolerance = group_avg_hours * (variance_pct / 100)
                    st.caption(f"Group average: {group_avg_hours:.1f} hrs/week | On-target range: {group_avg_hours - tolerance:.1f}–{group_avg_hours + tolerance:.1f} hrs/week (±{variance_pct}%)")
                else:
                    st.caption(f"Group average: {group_avg_hours:.1f} hrs/week | On-target range: exact match only (0% variance)")

                st.divider()

                st.divider()

                # Peer comparison vs department average
                st.subheader("vs. Department Average")

                dept_peer_df = filtered_df[filtered_df['Department'] == emp_data['Department']]
                compare_metrics = {
                    'Hours/Week': 'HoursSpentPerWeek',
                    'Total Cases': 'TotalCases',
                    'Meetings': 'UniqueMeetings',
                    'Calls': 'UniqueCalls',
                    'Emails': 'EmailCount',
                    'Hit Rate %': 'HitRatePct',
                    'Accounts Covered': 'AccountsCovered',
                    'Users Covered': 'UsersCovered',
                }

                compare_rows = []
                for label, col in compare_metrics.items():
                    dept_avg = float(dept_peer_df[col].mean() or 0)
                    emp_val = float(emp_data.get(col, 0) or 0)
                    pct_of_avg = (emp_val / dept_avg * 100) if dept_avg else 0
                    compare_rows.append({'Metric': label, 'PctOfAvg': pct_of_avg, 'EmpVal': emp_val, 'DeptAvg': dept_avg})
                compare_df = pd.DataFrame(compare_rows).sort_values('Metric', ascending=False)

                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    y=compare_df['Metric'],
                    x=compare_df['PctOfAvg'],
                    orientation='h',
                    marker=dict(color=[STATUS['good'] if v >= 100 else STATUS['critical'] for v in compare_df['PctOfAvg']]),
                    text=[f"{v:.0f}%" for v in compare_df['PctOfAvg']],
                    textposition='auto',
                    customdata=compare_df[['EmpVal', 'DeptAvg']].values,
                    hovertemplate='<b>%{y}</b><br>%{x:.0f}% of dept avg<br>Employee: %{customdata[0]:.1f} | Dept Avg: %{customdata[1]:.1f}<extra></extra>'
                ))
                fig_compare.add_vline(x=100, line_dash='dash', line_color='gray')
                fig_compare.update_layout(
                    xaxis_title='% of Department Average (100% = dept avg)',
                    yaxis_title='',
                    height=350,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_compare, use_container_width=True)

                st.divider()

                # Case & activity detail
                st.subheader("Case & Activity Detail")
                col1, col2, col3 = st.columns(3)
                with col1:
                    with st.container(border=True):
                        st.metric("Cases Created", f"{emp_data.get('CasesCreated', 0):,.0f}")
                with col2:
                    with st.container(border=True):
                        st.metric("Cases Delegated", f"{emp_data.get('CasesDelegated', 0):,.0f}")
                with col3:
                    with st.container(border=True):
                        st.metric("Cases Received", f"{emp_data.get('CasesReceived', 0):,.0f}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    with st.container(border=True):
                        st.metric("Call Hit Rate", f"{emp_data.get('HitRatePct', 0):.1f}%")
                with col2:
                    with st.container(border=True):
                        st.metric("Connections Made", f"{emp_data.get('ConnectionsMade', 0):,.0f}")
                with col3:
                    with st.container(border=True):
                        st.metric("Unique Clients Met", f"{emp_data.get('UniqueClientsMetWith', 0):,.0f}")

                st.divider()

                # Activity breakdown
                emp_activities = {
                    'Cases': float(emp_data.get('TotalCaseHours', 0) or 0),
                    'Meetings': float(emp_data.get('MeetingHours', 0) or 0),
                    'Calls': float(emp_data.get('CallTimeSpentHours', 0) or 0),
                    'Emails': float(emp_data.get('EmailHours', 0) or 0),
                    'RPDs': float(emp_data.get('RPD_EstHours', 0) or 0)
                }
                emp_activities = {k: v for k, v in emp_activities.items() if v > 0}

                if emp_activities:
                    total_emp_activity_hours = sum(emp_activities.values())
                    sorted_activities = sorted(emp_activities.items(), key=lambda x: x[1])
                    labels = [k for k, v in sorted_activities]
                    values = [v for k, v in sorted_activities]
                    colors_sorted = [ACTIVITY_COLORS.get(l, '#999999') for l in labels]

                    fig_emp_bar = go.Figure(data=[go.Bar(
                        y=labels,
                        x=values,
                        orientation='h',
                        marker=dict(color=colors_sorted),
                        text=[f"{v:,.1f}h ({v / total_emp_activity_hours * 100:.0f}%)" for v in values],
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>%{x:,.1f}h<extra></extra>'
                    )])
                    fig_emp_bar.update_layout(
                        title="Activity Breakdown",
                        xaxis_title="Hours",
                        yaxis_title="",
                        height=350,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.caption("ℹ️ Cases & Meetings hours are actual tracked time. Calls, Emails, and RPDs hours are estimates — see the Methodology tab.")
                    st.plotly_chart(fig_emp_bar, use_container_width=True)
        else:
            st.info("No employees in selected department(s)")
    else:
        st.info("No data available")

# ============================================================================
# TAB 4: TRENDS
# ============================================================================

with tab4:
    st.subheader("Activity Trends")
    st.info("🚧 Work in Progress")
    st.caption("ℹ️ Total Hours includes estimated time for Calls, Emails, and RPDs — see the Methodology tab for how each is calculated.")

    # Granularity selector
    col1, col2 = st.columns([1, 3])
    with col1:
        granularity = st.radio("Granularity", ["Weekly", "Monthly", "Quarterly"], horizontal=True, index=1)
    granularity_key = {"Weekly": "week", "Monthly": "month", "Quarterly": "quarter"}[granularity]

    if granularity == "Quarterly":
        st.caption("ℹ️ Fiscal quarters: Q1 Sep–Nov, Q2 Dec–Feb, Q3 Mar–May, Q4 Jun–Aug")

    # Load trend data
    trend_df = load_trend_data(fy_start, fy_end, granularity_key)

    if trend_df is not None and len(trend_df) > 0:
        trend_df['Period'] = pd.to_datetime(trend_df['Period'])
        trend_df = trend_df.sort_values('Period')

        # Respect the sidebar's Region/Department filters
        if selected_regions:
            trend_df = trend_df[trend_df['Region'].isin(selected_regions)]
        if selected_dept:
            trend_df = trend_df[trend_df['Department'].isin(selected_dept)]

    else:
        trend_df = trend_df.iloc[0:0] if trend_df is not None else pd.DataFrame()

    if trend_df is None or len(trend_df) == 0:
        st.warning("No trend data available for the current date range and Region/Department filter selection.")
    else:
        # Trend summary widgets: period-over-period growth, trend direction, busiest period
        org_totals = trend_df.groupby('Period').agg({
            'TotalHours': 'sum',
            'TotalCases': 'sum',
            'UniqueMeetings': 'sum'
        }).reset_index().sort_values('Period')

        def format_period_label(period):
            if granularity_key == 'week':
                return f"Week of {period.strftime('%b %d, %Y')}"
            elif granularity_key == 'quarter':
                quarter_map = {9: 1, 10: 1, 11: 1, 12: 2, 1: 2, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8: 4}
                return f"FY Q{quarter_map.get(period.month, '?')} ({period.strftime('%b %Y')})"
            return period.strftime('%b %Y')

        def pct_change(new, old):
            return ((new - old) / old * 100) if old else None

        st.write("**Trend Summary**")

        if len(org_totals) >= 2:
            latest = org_totals.iloc[-1]
            prior = org_totals.iloc[-2]
            hours_chg = pct_change(latest['TotalHours'], prior['TotalHours'])
            cases_chg = pct_change(latest['TotalCases'], prior['TotalCases'])
            meetings_chg = pct_change(latest['UniqueMeetings'], prior['UniqueMeetings'])

            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container(border=True):
                    st.metric(
                        f"Hours — {format_period_label(latest['Period'])}",
                        f"{latest['TotalHours']:,.0f}h",
                        delta=f"{hours_chg:+.1f}% vs prior period" if hours_chg is not None else None
                    )
            with col2:
                with st.container(border=True):
                    st.metric(
                        f"Cases — {format_period_label(latest['Period'])}",
                        f"{latest['TotalCases']:,.0f}",
                        delta=f"{cases_chg:+.1f}% vs prior period" if cases_chg is not None else None
                    )
            with col3:
                with st.container(border=True):
                    st.metric(
                        f"Meetings — {format_period_label(latest['Period'])}",
                        f"{latest['UniqueMeetings']:,.0f}",
                        delta=f"{meetings_chg:+.1f}% vs prior period" if meetings_chg is not None else None
                    )
        else:
            st.info("Need at least 2 periods of data to show period-over-period growth.")

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                if len(org_totals) >= 4:
                    half = len(org_totals) // 2
                    first_half_avg = org_totals.iloc[:half]['TotalHours'].mean()
                    second_half_avg = org_totals.iloc[half:]['TotalHours'].mean()
                    shift_pct = pct_change(second_half_avg, first_half_avg) or 0
                    if shift_pct > 5:
                        trend_label = "📈 Trending Up"
                    elif shift_pct < -5:
                        trend_label = "📉 Trending Down"
                    else:
                        trend_label = "➡️ Flat"
                    st.metric("Trend Direction", trend_label, help=f"{shift_pct:+.1f}% — second half of range vs first half of range (Total Hours)")
                else:
                    st.metric("Trend Direction", "N/A", help="Need at least 4 periods to determine trend direction")
        with col2:
            with st.container(border=True):
                peak_row = org_totals.loc[org_totals['TotalHours'].idxmax()]
                st.metric("Busiest Period", format_period_label(peak_row['Period']), help=f"{peak_row['TotalHours']:,.0f}h — highest Total Hours in the selected range")

        st.divider()

        # Section 2: Per-activity breakdown (stacked area)
        st.write("**Hours by Activity Type**")
        st.caption("ℹ️ Cases & Meetings are actual tracked time. Calls, Emails, and RPDs are estimates.")

        activity_hour_cols = {
            'Cases': 'CaseHours',
            'Meetings': 'MeetingHours',
            'Calls': 'CallHours',
            'Emails': 'EmailHours',
            'RPDs': 'RPDHours',
        }
        activity_trend = trend_df.groupby('Period')[list(activity_hour_cols.values())].sum().reset_index()

        fig_stack = go.Figure()
        for activity, col_name in activity_hour_cols.items():
            fig_stack.add_trace(go.Bar(
                x=activity_trend['Period'],
                y=activity_trend[col_name],
                name=activity,
                marker=dict(color=ACTIVITY_COLORS.get(activity, '#999999')),
                hovertemplate=f'<b>{activity}</b><br>%{{x|%b %d, %Y}}: %{{y:.0f}}h<extra></extra>'
            ))

        fig_stack.update_layout(
            height=400,
            barmode='stack',
            hovermode='x unified',
            xaxis_title='Period',
            yaxis_title='Hours',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        st.divider()

        # Section 4: Department breakdown
        st.write("**Trends by Department**")

        dept_options = sorted(trend_df['Department'].unique())
        dept_default = [d for d in selected_dept if d in dept_options] or dept_options[:3]

        dept_filter = st.multiselect(
            "Departments to display",
            options=dept_options,
            default=dept_default,
            key="trend_dept_filter"
        )

        if len(dept_filter) > 5:
            st.caption(f"Showing first 5 of {len(dept_filter)} selected departments for clarity — narrow the filter to see others")
            dept_filter = dept_filter[:5]

        if dept_filter:
            dept_trend = trend_df[trend_df['Department'].isin(dept_filter)]

            fig_dept = go.Figure()

            for i, dept in enumerate(dept_filter):
                dept_data = dept_trend[dept_trend['Department'] == dept].sort_values('Period')
                fig_dept.add_trace(go.Bar(
                    x=dept_data['Period'],
                    y=dept_data['TotalHours'],
                    name=dept,
                    marker=dict(color=CATEGORICAL[i % len(CATEGORICAL)]),
                    hovertemplate='<b>%{x|%b %d, %Y}</b><br>%{fullData.name}: %{y:.0f}h<extra></extra>'
                ))

            fig_dept.update_layout(
                height=400,
                barmode='group',
                hovermode='x unified',
                xaxis_title='Period',
                yaxis_title='Hours',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_dept, use_container_width=True)

        st.divider()

        st.subheader("Trend Data Table")
        st.dataframe(
            trend_df.sort_values('Period', ascending=False),
            use_container_width=True,
            height=400
        )

# ============================================================================
# TAB 5: DATA & EXPORT
# ============================================================================

with tab5:
    st.subheader("Detailed Data")

    if len(filtered_df) > 0:
        display_cols = [col for col in filtered_df.columns if col not in ['emp_label']]
        st.dataframe(filtered_df[display_cols], use_container_width=True, height=400)
    else:
        st.info("No employees in selected department(s).")

    st.divider()
    st.subheader("Export Data")

    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"workforce_analytics_{fy_start}_{fy_end}.csv",
        mime="text/csv"
    )

# ============================================================================
# TAB 6: METHODOLOGY
# ============================================================================

with tab6:
    st.subheader("Methodology & FAQ")
    st.write("How every metric in this dashboard is calculated, including which numbers are actual tracked time vs. estimates.")

    st.divider()
    st.markdown("### Source Tables")
    st.markdown("""
- `ea_prod.crmfacts_gold.cases` — Case/ticket data
- `ea_prod.crmfacts_gold.vmeetings` — Meeting activity data
- `ea_prod.crmfacts_gold.vactivity` — Call and email activity data
""")

    st.markdown("### Global Filters Applied")
    st.markdown("""
- **Date Filter:** `OpenedDate` / `MeetingDate` / `activity_date` within the selected date range
- **Team Filter:** `JobFamily = 'Client Consulting'` (all active Client Consulting departments)
""")

    st.divider()

    with st.expander("📁 Case Metrics", expanded=False):
        st.markdown("""
**Source:** `case_metrics` CTE, from `ea_prod.crmfacts_gold.cases`

**TotalCases**
- SQL: `COUNT(*)`
- Logic: Count all cases where person is the `CaseOwner`
- Group By: `CaseOwner`

**CasesCreated**
- Source CTE: `delegation_metrics`
- SQL: `COUNT(*)`
- Logic: Count all cases where person is the `CreatedBy`
- Group By: `CreatedBy`
- Join: Filtered to only team members via `INNER JOIN team_members`

**CasesDelegated**
- Source CTE: `delegation_metrics`
- SQL: `SUM(CASE WHEN c.CreatedBy != c.CaseOwner THEN 1 ELSE 0 END)`
- Logic: Count cases where `CreatedBy` = person BUT `CaseOwner` ≠ person (delegated away)

**CasesReceived**
- Source CTE: `delegation_combined`
- SQL: `(CurrentCasesOwned - (TotalCasesCreated - CasesDelegatedAway))`
- Formula: `CasesReceived = TotalCases - (CasesCreated - CasesDelegated)`

**TotalCaseHours** — *actual tracked time*
- Source CTE: `case_metrics`
- SQL: `ROUND(SUM(CAST(TimeToResolutionHrs AS DOUBLE)), 2)`
- Field Used: `TimeToResolutionHrs` (string converted to double)
""")

    with st.expander("📅 Meeting Metrics", expanded=False):
        st.markdown("""
**Source:** `meeting_metrics` CTE, from `ea_prod.crmfacts_gold.vmeetings`

**Data Quality Filter:** `WHERE DurationInMinutes <= 480` — filters out meetings > 8 hours (data quality issue)

**Deduplication — two levels**, to avoid row multiplication when joining attendee counts to meeting durations:

- *Dedup Level 1* — one row per `(MeetingId, EmployeeName)` via `meeting_dedup` → `unique_meetings`. Used for `UniqueMeetings` and `MeetingHours`.
- *Dedup Level 2* — one row per `(MeetingId, EmployeeName, IndividualId)` via `attendee_dedup` → `unique_attendees` (with `IndividualId IS NOT NULL`). Used for `UniqueClientsMetWith` and `UsersPerMeeting`.

The two levels are aggregated independently into `meeting_counts` and `attendee_counts`, then joined on `EmployeeName`. This prevents meeting durations from being multiplied by the number of attendees.

**UniqueMeetings** — *actual tracked time*
- SQL: `COUNT(DISTINCT um.MeetingId)` from `meeting_counts` (Dedup Level 1)

**UniqueClientsMetWith**
- SQL: `COUNT(DISTINCT ua.IndividualId)` from `attendee_counts` (Dedup Level 2)
- Logic: Distinct client contacts met with at least once. A client in 10 different meetings counts as 1. Only non-null `IndividualId` counted.

**UsersPerMeeting**
- SQL: `ROUND(TotalAttendeeSlots * 1.0 / NULLIF(UniqueMeetings, 0), 2)` where `TotalAttendeeSlots = COUNT(ua.IndividualId)`
- Logic: True average client attendees per meeting. Numerator is total (meeting, client) pairs, not distinct — one client in 3 meetings contributes 3.

**MeetingHours** — *actual tracked time*
- SQL: `ROUND(SUM(um.DurationInMinutes) / 60, 2)` from `meeting_counts` (Dedup Level 1)
""")

    with st.expander("📞 Call Metrics — ESTIMATED", expanded=False):
        st.markdown("""
**Source:** `call_email_metrics` CTE, from `ea_prod.crmfacts_gold.vactivity`

**Activity Type Filter:** `WHERE activity_type IN ('Call', 'Email')`

**Deduplication:** `deduplicated_activities` CTE — `ROW_NUMBER() OVER (PARTITION BY id, activity_owner_id ORDER BY id)` removes duplicate activity records

**Employee Matching:** `INNER JOIN case_owners ON a.EmployeeId = co.CaseOwnerId` — matches activity owner ID to case owner ID

**UniqueCalls**
- SQL: `COUNT(DISTINCT CASE WHEN a.ActivityType = 'Call' THEN a.ActivityId END)`

**ConnectionsMade**
- SQL: `SUM(CASE WHEN a.ActivityType = 'Call' AND a.ConnectionOutcome = 'Connection Made' THEN 1 ELSE 0 END)`

**HitRatePct**
- Formula: `(ConnectionsMade ÷ UniqueCalls) × 100`
- `NULLIF` prevents division by zero

**CallTimeSpentHours — ⚠️ ESTIMATE, not actual tracked time**
- Formula: `((Missed Calls × 2 min) + (Connected Calls × 5 min)) ÷ 60`
- Assumption: missed calls = 2 minutes each, connected calls = 5 minutes each
""")

    with st.expander("✉️ Email Metrics — ESTIMATED", expanded=False):
        st.markdown("""
**Source:** `call_email_metrics` CTE, from `ea_prod.crmfacts_gold.vactivity`

**EmailCount**
- SQL: `COUNT(DISTINCT CASE WHEN a.ActivityType = 'Email' THEN a.ActivityId END)`

**EmailHours — ⚠️ ESTIMATE, not actual tracked time**
- Formula: `(EmailCount × 10 min) ÷ 60`
- Assumption: each email = 10 minutes
""")

    with st.expander("📋 RPD Metrics — ESTIMATED", expanded=False):
        st.markdown("""
**Source:** `ea_prod.rpdfacts_gold.f_tr_rpd_comments` (comments) and `ea_prod.reporting_gold.pbi_rpd` (authored)

**RPDs_Authored**
- SQL: `COUNT(DISTINCT` `RPD#``)` from `rpd_authored_agg`, matched via `Author ID`

**RPD_Comments**
- SQL: `COUNT(*)` from `rpd_comments_agg`, matched via `CommenterId`

**RPD_EstHours — ⚠️ ESTIMATE, not actual tracked time**
- Formula: `(RPDs_Authored × 20 min + RPD_Comments × 10 min) ÷ 60`
- Assumption: 20 minutes per RPD authored, 10 minutes per comment made
""")

    with st.expander("⏱️ Aggregate Time Metrics", expanded=False):
        st.markdown("""
**TotalTimeSpentHours**
- Formula: `TotalCaseHours + MeetingHours + CallTimeSpentHours + EmailHours + RPD_EstHours`
- Uses `COALESCE` to treat NULLs as 0
- Mix of actual tracked time (Cases, Meetings) and estimates (Calls, Emails, RPDs)

**HoursSpentPerWeek**
- Formula: `TotalTimeSpentHours ÷ (Business Days in Period ÷ 5)`
- Logic: Average hours spent per week over the selected date range, using a Mon–Fri business-day calendar (federal holidays excluded)
- `NULLIF` prevents division by zero
""")

    with st.expander("🔗 Final Join Logic", expanded=False):
        st.markdown("""
```
FROM filtered_employees fe
LEFT JOIN case_metrics c        ON fe.FullName_FNF = c.CaseOwner
LEFT JOIN delegation_combined d ON fe.FullName_FNF = d.EmployeeName
LEFT JOIN meeting_metrics m     ON fe.FullName_FNF = m.EmployeeName
LEFT JOIN call_email_metrics ce ON fe.FullName_FNF = ce.EmployeeName
LEFT JOIN rpd_comments_agg rpd  ON fe.EmployeeID = rpd.EmployeeId
LEFT JOIN rpd_authored_agg ra   ON fe.EmployeeID = ra.AuthorId
```

- **Anchor:** All active Client Consulting employees (`filtered_employees`), so zero-activity employees are still included
- **Join Type:** `LEFT JOIN` — every employee is included even with no cases/meetings/calls/emails
- **COALESCE:** Replaces NULL values with 0 for employees with no activity in a category
""")

# Info footer
st.sidebar.markdown("---")
st.sidebar.info(
    "**Note:** This dashboard is connected to Databricks for real-time data. "
    "Adjust the date range and department filter to see insights for your selection."
)
