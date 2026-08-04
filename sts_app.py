import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date as _date
import numpy as np
from databricks.sql import connect

st.set_page_config(page_title="Workforce Analytics", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Workforce Capacity & Activity Analytics")

# ============================================================================
# DATA LOADING & PREP
# ============================================================================

@st.cache_resource
def get_databricks_connection():
    """Create a persistent Databricks connection"""
    return connect(host=st.secrets.get("DATABRICKS_HOST"),
                   http_path=st.secrets.get("DATABRICKS_HTTP_PATH"),
                   auth_type="oauth")

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
                CAST(`Author ID` AS INT) as AuthorId,
                COUNT(DISTINCT `RPD#`) as RPDs_Authored
              FROM ea_prod.reporting_gold.pbi_rpd
              WHERE `Created Date` >= '{fy_start}' AND `Created Date` <= '{fy_end}'
              GROUP BY CAST(`Author ID` AS INT)
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
# SIDEBAR: FILTERS & DATE RANGE
# ============================================================================

st.sidebar.header("🔍 Filters & Settings")

# Date range picker
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2026, 9, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

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
    default=departments[:3] if len(departments) > 3 else departments
)

# View type
view_type = st.sidebar.radio("View", ["Team Summary", "Individual Employees"], horizontal=True)

# Filter by selected departments
if selected_dept:
    filtered_df = df[df['Department'].isin(selected_dept)].copy()
else:
    filtered_df = df.copy()

# ============================================================================
# MAIN METRICS & OVERVIEW
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

total_hours = filtered_df['TotalTimeSpentHours'].sum() if len(filtered_df) > 0 else 0
avg_hours_per_week = filtered_df['HoursSpentPerWeek'].mean() if len(filtered_df) > 0 else 0
total_employees = len(filtered_df)
total_cases = filtered_df['TotalCases'].sum() if len(filtered_df) > 0 else 0
total_meetings = filtered_df['UniqueMeetings'].sum() if len(filtered_df) > 0 else 0

with col1:
    st.metric("Total Hours", f"{total_hours:,.0f}h")
with col2:
    st.metric("Avg Hrs/Week", f"{avg_hours_per_week:.1f}h")
with col3:
    st.metric("Active Employees", f"{total_employees}")
with col4:
    st.metric("Total Cases", f"{total_cases:,.0f}")
with col5:
    st.metric("Total Meetings", f"{total_meetings:,.0f}")

# ============================================================================
# TIME ALLOCATION PIE CHART
# ============================================================================

st.subheader("⏱️ Time Allocation by Activity")

col1, col2 = st.columns([2, 1])

with col1:
    # Calculate total hours by activity
    activity_data = {
        'Cases': filtered_df['TotalCaseHours'].sum(),
        'Meetings': filtered_df['MeetingHours'].sum(),
        'Calls': filtered_df['CallTimeSpentHours'].sum(),
        'Emails': filtered_df['EmailHours'].sum(),
        'RPDs': filtered_df['RPD_EstHours'].sum()
    }

    # Remove zero values for cleaner chart
    activity_data = {k: v for k, v in activity_data.items() if v > 0}

    if activity_data:
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(activity_data.keys()),
            values=list(activity_data.values()),
            marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']),
            hovertemplate='<b>%{label}</b><br>Hours: %{value:,.0f}<br>%{percent}<extra></extra>'
        )])
        fig_pie.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Activity summary table
    if activity_data:
        st.write("**Activity Hours**")
        activity_df = pd.DataFrame({
            'Activity': list(activity_data.keys()),
            'Hours': list(activity_data.values())
        })
        activity_df['% of Total'] = (activity_df['Hours'] / activity_df['Hours'].sum() * 100).round(1)
        activity_df = activity_df.sort_values('Hours', ascending=False)
        st.dataframe(activity_df, use_container_width=True, hide_index=True)

# ============================================================================
# TEAM SUMMARY vs INDIVIDUAL VIEW
# ============================================================================

if view_type == "Team Summary":
    st.subheader("👥 Team Summary")

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
        # Bar chart: Total hours by team
        col1, col2 = st.columns(2)

        with col1:
            fig_team_hrs = px.bar(
                team_summary.reset_index(),
                x='Department',
                y='TotalTimeSpentHours',
                title='Total Hours by Department',
                labels={'TotalTimeSpentHours': 'Hours', 'Department': 'Department'},
                color='HoursSpentPerWeek',
                color_continuous_scale='Viridis'
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

        # Team table
        st.dataframe(team_summary, use_container_width=True)

else:
    st.subheader("👤 Individual Employee Details")

    if len(filtered_df) > 0:
        # Create employee labels
        filtered_df['emp_label'] = filtered_df['EmployeeName'] + ' (' + filtered_df['TenureYears'].astype(str) + ' yrs)'
        emp_labels = filtered_df['emp_label'].tolist()

        if len(emp_labels) > 0:
            selected_employee = st.selectbox("Select Employee", emp_labels)

            # Find the employee row that matches the label
            selected_row = filtered_df[filtered_df['emp_label'] == selected_employee]
            if len(selected_row) > 0:
                emp_data = selected_row.iloc[0]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Hours", f"{emp_data['TotalTimeSpentHours']:.1f}h")
                with col2:
                    st.metric("Hours/Week", f"{emp_data['HoursSpentPerWeek']:.1f}h")
                with col3:
                    st.metric("Title", emp_data.get('JobTitle', 'N/A'))
                with col4:
                    st.metric("Tenure (Yrs)", f"{emp_data.get('TenureYears', 'N/A')}")

                # Activity breakdown for this employee
                emp_activities = {
                    'Cases': emp_data.get('TotalCaseHours', 0),
                    'Meetings': emp_data.get('MeetingHours', 0),
                    'Calls': emp_data.get('CallTimeSpentHours', 0),
                    'Emails': emp_data.get('EmailHours', 0),
                    'RPDs': emp_data.get('RPD_EstHours', 0)
                }
                emp_activities = {k: v for k, v in emp_activities.items() if v > 0}

                if emp_activities:
                    fig_emp_pie = go.Figure(data=[go.Pie(
                        labels=list(emp_activities.keys()),
                        values=list(emp_activities.values()),
                        marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'])
                    )])
                    fig_emp_pie.update_layout(title=f"Activity Breakdown for {emp_data.get('EmployeeName', 'Employee')}", height=400)
                    st.plotly_chart(fig_emp_pie, use_container_width=True)
        else:
            st.info("No employees in selected department(s)")

# ============================================================================
# DETAILED DATA TABLE
# ============================================================================

st.subheader("📋 Detailed Data")

if len(filtered_df) > 0:
    st.dataframe(filtered_df, use_container_width=True, height=400)
else:
    st.info("No employees in selected department(s).")

# ============================================================================
# DATA EXPORT
# ============================================================================

st.subheader("📥 Export Data")

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
    "💡 **Note:** This dashboard is connected to Databricks for real-time data. "
    "Adjust the date range above to refresh the analysis."
)
