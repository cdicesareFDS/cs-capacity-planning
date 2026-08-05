import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date as _date
import numpy as np
from databricks.sql import connect

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

@st.cache_resource
def get_databricks_connection():
    """Create a persistent Databricks connection"""
    import os

    # Check if running in Databricks App (env vars are set by Databricks)
    if "DATABRICKS_HOST" in os.environ and "DATABRICKS_TOKEN" in os.environ:
        # Databricks App: use environment variables set by runtime
        return connect(
            server_hostname=os.environ["DATABRICKS_HOST"],
            http_path="/sql/1.0/warehouses/5534359f9aac6560",
            access_token=os.environ["DATABRICKS_TOKEN"]
        )
    else:
        # Local dev: use OAuth with secrets
        return connect(
            server_hostname=st.secrets.get("DATABRICKS_HOST"),
            http_path=st.secrets.get("DATABRICKS_HTTP_PATH"),
            auth_type="oauth"
        )

@st.cache_data
def load_trend_data(fy_start, fy_end, granularity="month"):
    """Query Databricks for time-series trends (weekly or monthly)"""
    try:
        conn = get_databricks_connection()
        cursor = conn.cursor()

        # Build the query with date_trunc for bucketing
        query = f"""
            WITH filtered_employees AS (
              SELECT DISTINCT EmployeeID, FullName_FNF, JobTitle, Department, Active
              FROM ea_prod.reference_gold.employee
              WHERE JobFamily = 'Client Consulting' AND Active = 1
            ),
            case_trend AS (
              SELECT
                DATE_TRUNC('{granularity}', c.OpenedDate) as Period,
                fe.Department,
                COUNT(*) as TotalCases,
                SUM(CAST(c.TimeToResolutionHrs AS DOUBLE)) as CaseHours
              FROM ea_prod.crmfacts_gold.cases c
              INNER JOIN filtered_employees fe ON c.CaseOwnerId = fe.EmployeeID
              WHERE c.OpenedDate >= '{fy_start}' AND c.OpenedDate <= '{fy_end}'
              GROUP BY DATE_TRUNC('{granularity}', c.OpenedDate), fe.Department
            ),
            meeting_trend AS (
              SELECT
                DATE_TRUNC('{granularity}', v.MeetingDate) as Period,
                fe.Department,
                COUNT(DISTINCT v.MeetingId) as UniqueMeetings,
                ROUND(SUM(v.DurationInMinutes) / 60, 2) as MeetingHours
              FROM ea_prod.crmfacts_gold.vmeetings v
              INNER JOIN filtered_employees fe ON v.EmployeeName = fe.FullName_FNF
              WHERE v.MeetingDate >= '{fy_start}' AND v.MeetingDate <= '{fy_end}'
                AND v.DurationInMinutes <= 480
              GROUP BY DATE_TRUNC('{granularity}', v.MeetingDate), fe.Department
            ),
            call_email_trend AS (
              SELECT
                DATE_TRUNC('{granularity}', a.activity_date) as Period,
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
              GROUP BY DATE_TRUNC('{granularity}', a.activity_date), fe.Department
            )
            SELECT
              COALESCE(c.Period, m.Period, e.Period) as Period,
              COALESCE(c.Department, m.Department, e.Department) as Department,
              COALESCE(c.TotalCases, 0) as TotalCases,
              COALESCE(c.CaseHours, 0) as CaseHours,
              COALESCE(m.UniqueMeetings, 0) as UniqueMeetings,
              COALESCE(m.MeetingHours, 0) as MeetingHours,
              COALESCE(e.CallCount, 0) as CallCount,
              COALESCE(e.CallHours, 0) as CallHours,
              COALESCE(e.EmailCount, 0) as EmailCount,
              COALESCE(e.EmailHours, 0) as EmailHours,
              ROUND(
                COALESCE(c.CaseHours, 0) + COALESCE(m.MeetingHours, 0) +
                COALESCE(e.CallHours, 0) + COALESCE(e.EmailHours, 0), 2
              ) as TotalHours
            FROM case_trend c
            FULL OUTER JOIN meeting_trend m ON c.Period = m.Period AND c.Department = m.Department
            FULL OUTER JOIN call_email_trend e ON c.Period = e.Period AND c.Department = e.Department
            ORDER BY Period, Department
        """

        cursor.execute(query)
        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()

        return df
    except Exception as e:
        st.error(f"Error loading trend data from Databricks: {e}")
        return None

@st.cache_data
def load_data(fy_start, fy_end):
    """Query Databricks for combined metrics"""
    try:
        conn = get_databricks_connection()
        cursor = conn.cursor()

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
                SELECT DISTINCT EmployeeID, FullName_FNF, JobTitle, Department
                FROM ea_prod.reference_gold.employee
                WHERE JobFamily = 'Client Consulting' AND Active = 1
              ) fe ON c.CaseOwnerId = fe.EmployeeID
              WHERE c.OpenedDate >= '{fy_start}' AND c.OpenedDate <= '{fy_end}'
              GROUP BY fe.Department, c.CaseOwner
            ),
            filtered_employees AS (
              SELECT DISTINCT EmployeeID, FullName_FNF, JobTitle, Department, Active
              FROM ea_prod.reference_gold.employee
              WHERE JobFamily = 'Client Consulting' AND Active = 1
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
            LEFT JOIN ea_dev.reporting_gold.sl_employeedepartment_history hd
              ON fe.EmployeeID = hd.EmployeeId AND hd.Current = true
            LEFT JOIN case_metrics c        ON fe.FullName_FNF = c.CaseOwner
            LEFT JOIN delegation_combined d ON fe.FullName_FNF = d.EmployeeName
            LEFT JOIN meeting_metrics m     ON fe.FullName_FNF = m.EmployeeName
            LEFT JOIN call_email_metrics ce ON fe.FullName_FNF = ce.EmployeeName
            LEFT JOIN rpd_comments_agg rpd ON fe.EmployeeID = rpd.EmployeeId
            LEFT JOIN rpd_authored_agg ra  ON fe.EmployeeID = ra.AuthorId
            ORDER BY fe.Department, COALESCE(c.TotalCases, 0) DESC
        """

        cursor.execute(query)
        df = cursor.fetchall_arrow().to_pandas()
        cursor.close()

        return df
    except Exception as e:
        st.error(f"Error loading data from Databricks: {e}")
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

# Get unique departments
departments = sorted(df['Department'].unique().tolist())

# Department filter
selected_dept = st.sidebar.multiselect(
    "Department(s)",
    options=departments,
    default=[]
)

# Filter by selected departments
if selected_dept:
    filtered_df = df[df['Department'].isin(selected_dept)].copy()
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Team", "Individual", "Trends", "Data & Export"])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    st.subheader("Executive Summary")

    # KPI row with cards
    col1, col2, col3, col4, col5 = st.columns(5)

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
            st.metric("Total Cases", f"{total_cases:,.0f}")

    with col5:
        with st.container(border=True):
            st.metric("Total Meetings", f"{total_meetings:,.0f}")

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

    metric_cols = st.columns(len(activity_df))
    for idx, (col, (_, row)) in enumerate(zip(metric_cols, activity_df.iterrows())):
        with col:
            with st.container(border=True):
                st.metric(row['Activity'], f"{row['Hours']:,.0f}h")
                st.caption(f"{row['% of Total']:.0f}% of total")

    st.divider()

    # Detailed employee table by department (with department averages embedded)
    st.subheader("Employee Metrics by Department")

    for dept in sorted(filtered_df['Department'].unique()):
        dept_employees = filtered_df[filtered_df['Department'] == dept].sort_values('HoursSpentPerWeek', ascending=False)

        display_cols = ['EmployeeName', 'JobTitle', 'TenureYears', 'HoursSpentPerWeek',
                       'TotalCases', 'UniqueMeetings', 'UniqueCalls', 'HitRatePct', 'EmailCount', 'RPDs_Authored',
                       'CasesPct', 'MeetingsPct', 'CallsPct', 'EmailsPct', 'RPDsPct']

        # Filter to only columns that exist
        display_cols = [col for col in display_cols if col in dept_employees.columns]

        with st.expander(f"📂 {dept} ({len(dept_employees)} employees)", expanded=False):
            # Create department average row
            dept_avg_row = {}
            for col in display_cols:
                if col == 'EmployeeName':
                    dept_avg_row[col] = 'DEPT AVG'
                elif col == 'JobTitle':
                    dept_avg_row[col] = ''
                elif dept_employees[col].dtype in ['int64', 'float64']:
                    dept_avg_row[col] = dept_employees[col].mean()
                else:
                    dept_avg_row[col] = ''

            # Display department average separately (pinned)
            dept_avg_df = pd.DataFrame([dept_avg_row])

            # Employee data only
            display_df = dept_employees[display_cols].copy()

            def highlight_vs_dept_avg(val, col_name, df):
                """Color a cell based on its variance from department average"""
                if pd.isna(val) or not isinstance(val, (int, float)):
                    return ''

                col_data = df[col_name]
                dept_avg = col_data.iloc[0]  # First row is the average

                if dept_avg == 0 or val == 0:
                    return ''

                variance_pct = ((val - dept_avg) / dept_avg) * 100

                if variance_pct >= 15:  # GREEN - above dept average
                    intensity = min(abs(variance_pct) / 100, 1)
                    return f"background-color: rgba(12, 163, 12, {intensity * 0.7})"
                elif variance_pct <= -15:  # RED - below dept average
                    intensity = min(abs(variance_pct) / 100, 1)
                    return f"background-color: rgba(211, 59, 59, {intensity * 0.7})"
                else:  # YELLOW - within ±15% of dept average
                    return "background-color: rgba(250, 178, 25, 0.5)"

            # Format department average
            pct_cols = ['CasesPct', 'MeetingsPct', 'CallsPct', 'EmailsPct', 'RPDsPct']
            format_dict = {}
            for col in dept_avg_df.columns:
                if col in pct_cols:
                    format_dict[col] = lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else x
                elif col == 'HoursSpentPerWeek':
                    format_dict[col] = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else x
                elif col == 'HitRatePct':
                    format_dict[col] = lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else x
                elif col not in ['EmployeeName', 'JobTitle']:
                    if dept_avg_df[col].dtype in ['int64', 'float64']:
                        format_dict[col] = lambda x: f"{int(x)}" if isinstance(x, (int, float)) and not pd.isna(x) else x

            styled_avg = dept_avg_df.style.format(format_dict).set_properties(**{'background-color': '#f0f2f6', 'font-weight': 'bold'})
            st.dataframe(
                styled_avg,
                use_container_width=True,
                hide_index=True
            )

            # Apply styling to employee data BEFORE formatting
            styled = display_df.style
            for col in display_df.columns:
                if display_df[col].dtype in ['int64', 'float64']:
                    styled = styled.applymap(
                        lambda x, c=col: highlight_vs_dept_avg(x, c, dept_avg_df),
                        subset=[col]
                    )

            # Now format employees for display
            format_dict_emp = {}
            for col in display_df.columns:
                if col in pct_cols:
                    format_dict_emp[col] = lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else x
                elif col == 'HoursSpentPerWeek':
                    format_dict_emp[col] = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else x
                elif col == 'HitRatePct':
                    format_dict_emp[col] = lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else x
                elif col not in ['EmployeeName', 'JobTitle']:
                    if display_df[col].dtype in ['int64', 'float64']:
                        format_dict_emp[col] = lambda x: f"{int(x)}" if isinstance(x, (int, float)) and not pd.isna(x) else x

            styled = styled.format(format_dict_emp)

            st.dataframe(
                styled,
                use_container_width=True,
                height=400,
                hide_index=True
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
            fig_team_hrs = px.bar(
                team_summary.reset_index(),
                x='Department',
                y='TotalTimeSpentHours',
                title='Total Hours by Department',
                labels={'TotalTimeSpentHours': 'Hours', 'Department': 'Department'},
                color='HoursSpentPerWeek',
                color_continuous_scale=SEQUENTIAL_BLUE
            )
            fig_team_hrs.update_layout(xaxis_tickangle=-45, height=400)
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

        # Heatmap: activity allocation % by department
        dept_activity_mix = filtered_df.groupby('Department')[['CasesPct', 'MeetingsPct', 'CallsPct', 'EmailsPct', 'RPDsPct']].mean()
        dept_activity_mix.columns = ['Cases', 'Meetings', 'Calls', 'Emails', 'RPDs']

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=dept_activity_mix.T.values,
            x=dept_activity_mix.index,
            y=['Cases', 'Meetings', 'Calls', 'Emails', 'RPDs'],
            colorscale=SEQUENTIAL_BLUE,
            hovertemplate='<b>%{y}</b> | %{x}<br>%{z:.1f}%<extra></extra>',
            text=np.round(dept_activity_mix.T.values, 1),
            texttemplate='%{text:.1f}%',
            textfont={"size": 10}
        ))
        fig_heatmap.update_layout(height=300, xaxis_tickangle=-45)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.divider()
        st.subheader("Team Summary Table")
        st.dataframe(team_summary, use_container_width=True)

# ============================================================================
# TAB 3: INDIVIDUAL
# ============================================================================

with tab3:
    st.subheader("Individual Employee Details")

    if len(filtered_df) > 0:
        filtered_df['emp_label'] = filtered_df['EmployeeName'] + ' (' + filtered_df['TenureYears'].astype(str) + ' yrs)'
        emp_labels = filtered_df['emp_label'].tolist()

        if len(emp_labels) > 0:
            selected_employee = st.selectbox("Select Employee", emp_labels)

            selected_row = filtered_df[filtered_df['emp_label'] == selected_employee]
            if len(selected_row) > 0:
                emp_data = selected_row.iloc[0]

                # Employee info with capacity badge
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    with st.container(border=True):
                        st.metric("Total Hours", f"{emp_data['TotalTimeSpentHours']:.1f}h")
                with col2:
                    with st.container(border=True):
                        st.metric("Hours/Week", f"{emp_data['HoursSpentPerWeek']:.1f}h")
                with col3:
                    with st.container(border=True):
                        st.metric("Title", emp_data.get('JobTitle', 'N/A'))
                with col4:
                    with st.container(border=True):
                        st.metric("Tenure (Yrs)", f"{emp_data.get('TenureYears', 'N/A')}")

                st.divider()

                # Capacity status
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"Capacity Status: {emp_data['CapacityStatus']}")
                    if variance_pct > 0:
                        tolerance = group_avg_hours * (variance_pct / 100)
                        st.write(f"Group average: {group_avg_hours:.1f} hrs/week | On-target range: {group_avg_hours - tolerance:.1f}–{group_avg_hours + tolerance:.1f} hrs/week (±{variance_pct}%)")
                    else:
                        st.write(f"Group average: {group_avg_hours:.1f} hrs/week | Variance: 0% (exact match only)")

                st.divider()

                # Activity breakdown
                emp_activities = {
                    'Cases': emp_data.get('TotalCaseHours', 0),
                    'Meetings': emp_data.get('MeetingHours', 0),
                    'Calls': emp_data.get('CallTimeSpentHours', 0),
                    'Emails': emp_data.get('EmailHours', 0),
                    'RPDs': emp_data.get('RPD_EstHours', 0)
                }
                emp_activities = {k: v for k, v in emp_activities.items() if v > 0}

                if emp_activities:
                    colors_list = [ACTIVITY_COLORS.get(activity, '#999999') for activity in emp_activities.keys()]
                    fig_emp_pie = go.Figure(data=[go.Pie(
                        labels=list(emp_activities.keys()),
                        values=list(emp_activities.values()),
                        marker=dict(colors=colors_list)
                    )])
                    fig_emp_pie.update_layout(title=f"Activity Breakdown", height=400)
                    st.plotly_chart(fig_emp_pie, use_container_width=True)
        else:
            st.info("No employees in selected department(s)")
    else:
        st.info("No data available")

# ============================================================================
# TAB 4: TRENDS
# ============================================================================

with tab4:
    st.subheader("Activity Trends")

    # Granularity selector
    col1, col2 = st.columns([1, 3])
    with col1:
        granularity = st.radio("Granularity", ["Weekly", "Monthly"], horizontal=True)
    granularity_key = "week" if granularity == "Weekly" else "month"

    # Load trend data
    trend_df = load_trend_data(fy_start, fy_end, granularity_key)

    if trend_df is not None and len(trend_df) > 0:
        trend_df['Period'] = pd.to_datetime(trend_df['Period'])
        trend_df = trend_df.sort_values('Period')

        # Option 1: Org-wide total (default view)
        st.write("**Organization-wide Total Hours**")

        org_total = trend_df.groupby('Period')['TotalHours'].sum().reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=org_total['Period'],
            y=org_total['TotalHours'],
            mode='lines+markers',
            name='Total Hours',
            line=dict(color=SEQUENTIAL_BLUE[3], width=2),
            fill='tozeroy',
            fillcolor=f'rgba({int(SEQUENTIAL_BLUE[3][1:3], 16)}, {int(SEQUENTIAL_BLUE[3][3:5], 16)}, {int(SEQUENTIAL_BLUE[3][5:7], 16)}, 0.1)',
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Hours: %{y:.0f}<extra></extra>'
        ))

        fig_trend.update_layout(
            height=400,
            hovermode='x unified',
            xaxis_title='Period',
            yaxis_title='Hours',
            showlegend=False
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.divider()

        # Option 2: Department breakdown
        st.write("**Trends by Department**")

        dept_filter = st.multiselect(
            "Departments to display",
            options=sorted(trend_df['Department'].unique()),
            default=sorted(trend_df['Department'].unique())[:3]
        )

        if len(dept_filter) > 5:
            st.caption(f"Showing first 5 of {len(dept_filter)} selected departments for clarity — narrow the filter to see others")
            dept_filter = dept_filter[:5]

        if dept_filter:
            dept_trend = trend_df[trend_df['Department'].isin(dept_filter)]

            fig_dept = go.Figure()

            for i, dept in enumerate(dept_filter):
                dept_data = dept_trend[dept_trend['Department'] == dept].sort_values('Period')
                fig_dept.add_trace(go.Scatter(
                    x=dept_data['Period'],
                    y=dept_data['TotalHours'],
                    mode='lines+markers',
                    name=dept,
                    line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2),
                    hovertemplate='<b>%{x|%b %d, %Y}</b><br>%{fullData.name}: %{y:.0f}h<extra></extra>'
                ))

            fig_dept.update_layout(
                height=400,
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
    else:
        st.warning("No trend data available for the selected date range")

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

# Info footer
st.sidebar.markdown("---")
st.sidebar.info(
    "**Note:** This dashboard is connected to Databricks for real-time data. "
    "Adjust the date range and department filter to see insights for your selection."
)
