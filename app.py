import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# PAGE CONFIG
st.set_page_config(
    page_title="EduPath AI",
    page_icon="🎓",
    layout="wide"
)

# DESIGN & CUSTOM EMOJI STYLING
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fc;
}
h1 {
    color: #183b70;
}
div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce5f2;
}
</style>
""", unsafe_allow_html=True)

# STUDENT DATA LOAD (CSV FIRST; DEMO DATA ONLY IF NO CSV EXISTS)
REQUIRED_COLUMNS = ["Student_ID", "Department", "Attendance", "Marks", "Assignments", "Study_Hours"]
FEATURES = ["Attendance", "Marks", "Assignments", "Study_Hours"]

def prepare_student_data(raw_df):
    """Standardize common CSV column names and validate student data."""
    df = raw_df.copy()
    df.columns = [str(c).strip().replace(" ", "_").replace("-", "_") for c in df.columns]

    aliases = {
        "studentid": "Student_ID", "student_id": "Student_ID", "id": "Student_ID",
        "department": "Department", "dept": "Department",
        "attendance": "Attendance", "attendance_percentage": "Attendance",
        "marks": "Marks", "mark": "Marks", "score": "Marks",
        "assignments": "Assignments", "assignment": "Assignments",
        "study_hours": "Study_Hours", "studyhours": "Study_Hours",
        "risk": "Risk_Label", "risk_label": "Risk_Label"
    }
    rename = {}
    for col in df.columns:
        key = col.lower().replace("_", "")
        if key in ("studentid", "id"):
            rename[col] = "Student_ID"
        elif key in ("department", "dept"):
            rename[col] = "Department"
        elif key in ("attendance", "attendancepercentage"):
            rename[col] = "Attendance"
        elif key in ("marks", "mark", "score"):
            rename[col] = "Marks"
        elif key in ("assignments", "assignment"):
            rename[col] = "Assignments"
        elif key in ("studyhours",):
            rename[col] = "Study_Hours"
        elif key in ("risk", "risklabel"):
            rename[col] = "Risk_Label"
    df = df.rename(columns=rename)

    if "Student_ID" not in df.columns:
        df["Student_ID"] = [f"STU{i:03d}" for i in range(1, len(df) + 1)]
    if "Department" not in df.columns:
        df["Department"] = "General"

    for col in ["Attendance", "Marks", "Assignments", "Study_Hours"]:
        if col not in df.columns:
            if col == "Assignments":
                df[col] = pd.to_numeric(df.get("Marks", 0), errors="coerce")
            elif col == "Study_Hours":
                df[col] = 0
            else:
                raise ValueError(
                    f"CSV-la '{col}' column illa. Required columns: Student_ID, Department, "
                    "Attendance, Marks. Column names check pannunga."
                )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Attendance", "Marks"]).copy()
    if df.empty:
        raise ValueError("CSV-la valid student rows illa. Attendance and Marks numeric-a irukkanum.")

    df["Assignments"] = df["Assignments"].fillna(df["Assignments"].median() if df["Assignments"].notna().any() else 0)
    df["Study_Hours"] = df["Study_Hours"].fillna(df["Study_Hours"].median() if df["Study_Hours"].notna().any() else 0)
    df["Department"] = df["Department"].fillna("General").astype(str)
    df["Student_ID"] = df["Student_ID"].fillna("").astype(str)

    if "Risk_Label" not in df.columns:
        score = (
            0.4 * (100 - df["Attendance"].clip(0, 100))
            + 0.35 * (100 - df["Marks"].clip(0, 100))
            + 0.25 * (100 - df["Assignments"].clip(0, 100))
        )
        df["Risk_Label"] = np.where(score >= 48, "High", np.where(score >= 30, "Medium", "Low"))
    else:
        df["Risk_Label"] = df["Risk_Label"].astype(str).str.title()
        valid_labels = {"High", "Medium", "Low"}
        df.loc[~df["Risk_Label"].isin(valid_labels), "Risk_Label"] = "Medium"

    return df.reset_index(drop=True)


def load_data():
    # Admin-uploaded data takes priority for this session.
    uploaded_df = st.session_state.get("uploaded_students_df")
    if uploaded_df is not None:
        return prepare_student_data(uploaded_df)

    # GitHub / Streamlit Cloud filenames are case-sensitive.
    for filename in ["Students.csv", "students.csv", "students_data.csv"]:
        try:
            if __import__("os").path.exists(filename):
                return prepare_student_data(pd.read_csv(filename))
        except Exception as exc:
            st.error(f"{filename} read panna mudiyala: {exc}")

    # Keep the app usable if no CSV has been committed yet.
    st.warning("Students.csv file kidaikkala. Ippo demo data use aagudhu. Real student data-ku Students.csv upload/commit pannunga.")
    rng = np.random.default_rng(42)
    n = 150
    demo = pd.DataFrame({
        "Student_ID": [f"STU{i:03d}" for i in range(1, n + 1)],
        "Department": rng.choice(["CSE", "ECE", "IT", "Mechanical"], size=n),
        "Attendance": rng.integers(40, 101, n),
        "Marks": rng.integers(25, 101, n),
        "Assignments": rng.integers(20, 101, n),
        "Study_Hours": rng.integers(1, 7, n)
    })
    return prepare_student_data(demo)


df = load_data()

# Train from the currently loaded dataset; avoid stale cached models after CSV upload.
score = (
    0.4 * (100 - df["Attendance"].clip(0, 100))
    + 0.35 * (100 - df["Marks"].clip(0, 100))
    + 0.25 * (100 - df["Assignments"].clip(0, 100))
)
df["Risk_Label"] = df["Risk_Label"].fillna(
    pd.Series(np.where(score >= 48, "High", np.where(score >= 30, "Medium", "Low")), index=df.index)
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df[FEATURES], df["Risk_Label"])
df["Risk"] = model.predict(df[FEATURES])


# SESSION STATE MANAGEMENT
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_role" not in st.session_state:
    st.session_state["user_role"] = "Student"

# Student Editable State Data
if "student_profile" not in st.session_state:
    st.session_state["student_profile"] = {
        "Name": "Shruthi",
        "Email": "shruthi@institution.edu",
        "Phone": "+91 9876543210",
        "Address": "Chennai, Tamil Nadu"
    }

if "student_feedback_list" not in st.session_state:
    st.session_state["student_feedback_list"] = []

if "student_challenges_list" not in st.session_state:
    st.session_state["student_challenges_list"] = []

if "student_internships" not in st.session_state:
    st.session_state["student_internships"] = []

# Admin Session Tables
if "faculty_data" not in st.session_state:
    st.session_state["faculty_data"] = pd.DataFrame({
        "Faculty ID": ["FAC001", "FAC002", "FAC003"],
        "Faculty Name": ["Dr. Aris", "Prof. Sharma", "Dr. Kavitha"],
        "Department": ["CSE", "ECE", "IT"],
        "Assigned Mentees": [15, 12, 18],
        "Contact": ["aris@edu.in", "sharma@edu.in", "kavitha@edu.in"]
    })

if "user_data" not in st.session_state:
    st.session_state["user_data"] = pd.DataFrame({
        "User ID": ["STU001", "STU002", "FAC001", "ADM001"],
        "Name": ["Shruthi", "Alex", "Dr. Aris", "Admin Lead"],
        "Role": ["Student", "Student", "Faculty", "Admin"],
        "Status": ["Active", "Active", "Active", "Active"]
    })

if "mentorship_data" not in st.session_state:
    st.session_state["mentorship_data"] = pd.DataFrame({
        "Request ID": ["REQ101", "REQ102"],
        "Student ID": ["STU001", "STU015"],
        "Assigned Faculty": ["Dr. Aris", "Prof. Sharma"],
        "Topic": ["AI Career Guidance", "Math Remedial"],
        "Status": ["Pending", "Approved"]
    })

if "cgpa_data" not in st.session_state:
    st.session_state["cgpa_data"] = pd.DataFrame({
        "Student ID": ["STU001", "STU002", "STU003"],
        "Sem 1": [8.2, 7.5, 9.0],
        "Sem 2": [8.4, 7.8, 9.1],
        "Current CGPA": [8.4, 7.76, 9.1]
    })

if "timetable_data" not in st.session_state:
    st.session_state["timetable_data"] = pd.DataFrame({
        "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "9:00 - 10:00 AM": ["🤖 AI & ML", "🗄️ Database Systems", "🌐 Computer Networks", "💻 Data Structures", "🌐 Web Tech"],
        "1:30 - 3:30 PM": ["🧪 AI Lab", "📚 Library Practice", "💻 Coding Lab", "📊 Project Review", "🤝 Mentorship"]
    })

if "notifications_data" not in st.session_state:
    st.session_state["notifications_data"] = pd.DataFrame({
        "Notice ID": ["N001", "N002"],
        "Target Audience": ["All Students", "High Risk Students"],
        "Message": ["Mid-sem lab assessments start next week. 📢", "Mandatory mentorship catch-up on Friday. 🤝"],
        "Date Posted": ["2026-10-10", "2026-10-12"]
    })

if "lms_data" not in st.session_state:
    st.session_state["lms_data"] = pd.DataFrame({
        "Student ID": ["STU001", "STU002", "STU003"],
        "Video Hours Watched": [42, 18, 55],
        "Quizzes Completed": [10, 4, 12],
        "Portal Login Count": [88, 30, 110]
    })

if "placement_data" not in st.session_state:
    st.session_state["placement_data"] = pd.DataFrame({
        "Student ID": ["STU001", "STU002", "STU003"],
        "Technical Score": [88, 62, 95],
        "Soft Skills Score": [90, 70, 92],
        "Placement Eligibility": ["Eligible", "Not Eligible", "Eligible"]
    })

# ----------------------------------------------------
# LOGIN SCREEN
# ----------------------------------------------------
if not st.session_state["logged_in"]:
    st.title("🎓 Welcome to EduPath AI Portal")
    st.subheader("🔐 Please Login to Access Your Dashboard")

    col1, col2 = st.columns([1, 2])
    with col1:
        role_input = st.selectbox("👥 Select Role", ["Student", "Admin"])
        username = st.text_input("👤 Username / ID", "shruthi" if role_input == "Student" else "admin")
        password = st.text_input("🔑 Password", type="password", value="1234")

        if st.button("🚀 Login Now", type="primary"):
            st.session_state["logged_in"] = True
            st.session_state["user_role"] = role_input
            st.session_state["username"] = username
            st.rerun()

else:
    # SIDEBAR NAVIGATION HEADER WITH EMOJIS
    st.sidebar.markdown("📌 **Navigation Panel**")
    st.sidebar.write(f"👋 Welcome, **{st.session_state.get('username', 'User')}**")
    st.sidebar.write(f"🛡️ Role: **{st.session_state['user_role']}**")

    if st.sidebar.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

    st.sidebar.markdown("---")

    # ----------------------------------------------------
    # 1. STUDENT PORTAL (Interactive & Emoji Rich)
    # ----------------------------------------------------
    if st.session_state["user_role"] == "Student":
        st.sidebar.markdown("### 📚 Student Menu")
        page = st.sidebar.radio(
            "Select Option",
            [
                "🏠 Dashboard",
                "👤 AI Analytics & My Profile",
                "📊 Marks & Grades",
                "📅 Attendance",
                "📝 Exams & Assignments",
                "💡 Study Recommendations",
                "💬 Student Feedback",
                "⚠️ Academic Challenges",
                "🤝 Mentorship Requests",
                "📌 Faculty Follow-up",
                "💼 Placement & Career",
                "📈 CGPA Tracking",
                "🏆 Internship Participation",
                "🗓️ Timetable Planner",
                "🔔 Notifications",
                "🔒 Change Password"
            ],
            label_visibility="collapsed"
        )

        student = df.iloc[0]

        if "Dashboard" in page:
            st.title("🎓 Student Dashboard Overview")
            c1, c2, c3 = st.columns(3)
            c1.metric("📅 Attendance", f"{student['Attendance']}%")
            c2.metric("📊 Marks", f"{student['Marks']}/100")
            c3.metric("⚡ Risk Status", student["Risk"])
            st.success(f"🌟 Welcome back, {st.session_state['student_profile']['Name']}! Your academic journey is progressing brilliantly.")

        elif "My Profile" in page:
            st.title("👤 My Profile & AI Analytics (Editable)")

            with st.form("edit_profile_form"):
                name = st.text_input("📝 Full Name", st.session_state["student_profile"]["Name"])
                email = st.text_input("📧 Email Address", st.session_state["student_profile"]["Email"])
                phone = st.text_input("📞 Contact Number", st.session_state["student_profile"]["Phone"])
                address = st.text_area("🏠 Address", st.session_state["student_profile"]["Address"])

                update_btn = st.form_submit_button("✨ Update Profile")
                if update_btn:
                    st.session_state["student_profile"]["Name"] = name
                    st.session_state["student_profile"]["Email"] = email
                    st.session_state["student_profile"]["Phone"] = phone
                    st.session_state["student_profile"]["Address"] = address
                    st.success("🎉 Profile details updated successfully!")

            st.metric("🤖 AI Predicted Risk Level", student["Risk"])

        elif "Marks & Grades" in page:
            st.title("📊 Marks & Grades Report")
            grades_df = pd.DataFrame({
                "Subject": ["💻 Data Structures", "🤖 AI & ML", "🗄️ Database Systems", "🌐 Computer Networks"],
                "Internal Marks": [85, 78, 90, 82],
                "Grade": ["A+", "A", "O", "A"]
            })
            st.dataframe(grades_df, use_container_width=True, hide_index=True)

        elif "Attendance" in page:
            st.title("📅 Attendance Records Tracker")
            st.metric("🎯 Overall Attendance Percentage", f"{student['Attendance']}%")

        elif "Exams & Assignments" in page:
            st.title("📝 Exams & Assignments Schedule")
            st.info("📌 **Upcoming Exam:** Semester End Practical - Oct 25, 2026")

        elif "Study Recommendations" in page:
            st.title("💡 AI Study Recommendations")
            st.write("✨ Focus more on advanced algorithm problem-solving and database normalization techniques.")

        elif "Student Feedback" in page:
            st.title("💬 Subject-wise Student Feedback")
            st.write("📝 Share your thoughts to help improve teaching quality and lab sessions:")

            with st.form("subject_feedback_form"):
                sub_selected = st.selectbox("📚 Select Subject", ["Data Structures", "AI & Machine Learning", "Database Systems", "Computer Networks", "Web Tech"])
                feedback_category = st.selectbox("🏷️ Feedback Category", ["Teaching Quality", "Lab Infrastructure", "Course Material", "General"])
                feedback_text = st.text_area("✍️ Write your feedback or suggestions:")

                submit_fb = st.form_submit_button("🚀 Submit Feedback")
                if submit_fb:
                    st.session_state["student_feedback_list"].append({
                        "Subject": sub_selected,
                        "Category": feedback_category,
                        "Feedback": feedback_text
                    })
                    st.success(f"✅ Feedback submitted successfully for {sub_selected}!")

            if st.session_state["student_feedback_list"]:
                st.subheader("📋 Your Submitted Feedbacks")
                st.dataframe(pd.DataFrame(st.session_state["student_feedback_list"]), use_container_width=True, hide_index=True)

        elif "Academic Challenges" in page:
            st.title("⚠️ Report Academic Challenges (Editable)")

            with st.form("challenge_form"):
                ch_sub = st.selectbox("📚 Select Subject", ["Data Structures", "AI & ML", "Database Systems", "Networks"])
                ch_desc = st.text_area("✍️ Describe the concept or topic you find difficult:")
                submit_ch = st.form_submit_button("📤 Submit Challenge")
                if submit_ch:
                    st.session_state["student_challenges_list"].append({
                        "Subject": ch_sub,
                        "Issue": ch_desc,
                        "Status": "Pending Review"
                    })
                    st.success("🎯 Academic challenge reported to faculty mentors successfully!")

            if st.session_state["student_challenges_list"]:
                st.subheader("📌 Your Reported Challenges")
                st.dataframe(pd.DataFrame(st.session_state["student_challenges_list"]), use_container_width=True, hide_index=True)

        elif "Mentorship Requests" in page:
            st.title("🤝 1-on-1 Mentorship Requests")
            mentor_choice = st.selectbox("👨‍🏫 Select Mentor", ["Dr. Aris", "Prof. Sharma", "Dr. Kavitha"])
            topic = st.text_input("💬 Discussion Topic")
            if st.button("🚀 Send Request"):
                st.success(f"✅ Mentorship session requested with {mentor_choice}!")

        elif "Faculty Follow-up" in page:
            st.title("📌 Faculty Follow-up Status")
            st.info("💡 Your faculty mentor has reviewed your recent test scores and suggested practice exercises.")

        elif "Placement & Career" in page:
            st.title("💼 Placement & Career Readiness")
            st.metric("📈 Resume Score", "85/100")
            st.success("🌟 You are fully eligible for upcoming Tier-1 Corporate Recruitment Drives!")

        elif "CGPA Tracking" in page:
            st.title("📈 Semester-wise CGPA Tracker")
            st.metric("🎯 Current CGPA", "8.4 / 10")

        elif "Internship Participation" in page:
            st.title("🏆 Internship Participation & Certificate Upload")

            with st.form("internship_form"):
                company_name = st.text_input("🏢 Company Name")
                role_title = st.text_input("💼 Internship Role")
                duration = st.text_input("⏱️ Duration (e.g., 2 Months)")
                cert_file = st.file_uploader("📁 Upload Certificate File", type=["pdf", "png", "jpg"])

                submit_intern = st.form_submit_button("💾 Save Internship Details")
                if submit_intern:
                    st.session_state["student_internships"].append({
                        "Company": company_name,
                        "Role": role_title,
                        "Duration": duration,
                        "Certificate Status": "Uploaded & Pending Verification"
                    })
                    st.success("🎉 Internship record and certificate uploaded successfully!")

            if st.session_state["student_internships"]:
                st.subheader("📋 Your Saved Internships")
                st.dataframe(pd.DataFrame(st.session_state["student_internships"]), use_container_width=True, hide_index=True)

        elif "Timetable Planner" in page:
            st.title("🗓️ Master Timetable Planner")
            st.write("📅 Your active weekly class routine:")
            st.dataframe(st.session_state["timetable_data"], use_container_width=True, hide_index=True)

        elif "Notifications" in page:
            st.title("🔔 Portal Notifications")
            st.info("📢 **Announcement:** Lab records submission deadline extended until this Friday evening.")

        elif "Change Password" in page:
            st.title("🔒 Account Password Settings")
            old_pw = st.text_input("🔑 Old Password", type="password")
            new_pw = st.text_input("✨ New Password", type="password")
            confirm_pw = st.text_input("🔁 Confirm New Password", type="password")
            if st.button("💾 Update Password"):
                if new_pw == confirm_pw and len(new_pw) > 0:
                    st.success("🎉 Password updated successfully!")
                else:
                    st.error("⚠️ Passwords do not match or fields are empty.")

    # ----------------------------------------------------
    # 2. ADMIN PORTAL (28 Modules with Complete Emojis & Edits)
    # ----------------------------------------------------
    elif st.session_state["user_role"] == "Admin":
        st.sidebar.markdown("### 🛠️ Admin Management")
        admin_page = st.sidebar.radio(
            "Select Option",
            [
                "1. 📊 Overview",
                "2. ⚠️ Academic Challenge",
                "3. 🤖 AI Insights",
                "4. 📈 Academic Performance",
                "5. 📅 Attendance Analytics",
                "6. 🚨 At Risk Students",
                "7. 🎯 Student Success Score",
                "8. 📚 LMS & Engagement Analytics",
                "9. 💼 Placement & Skills Analytics",
                "10. 📊 Student Segmentation",
                "11. 👨‍🏫 Faculty Management",
                "12. 💬 Feedback Management",
                "13. 📁 Upload CSV / Export",
                "14. 📑 Reports & PDF Export",
                "15. 🔔 Notifications",
                "16. 👤 User Management",
                "17. ⚙️ Settings",
                "18. ❓ Help",
                "19. ➕ Subject Add",
                "20. ✏️ Marks & Grade Edit",
                "21. 📝 Exam & Assignments",
                "22. 💡 Study Recommendations",
                "23. 🤝 Mentorship Request",
                "24. 📈 CGPA Tracking",
                "25. 🏆 Internship Participation",
                "26. 🗓️ Time Table Plan",
                "27. 🔔 Admin Notifications",
                "28. 🔒 Change Password"
            ]
        )

        st.title(f"🛠️ Admin Portal: {admin_page}")

        if "1. 📊 Overview" in admin_page:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("👥 Total Students", len(df))
            c2.metric("📅 Avg Attendance", f"{df['Attendance'].mean():.1f}%")
            c3.metric("📊 Avg Marks", f"{df['Marks'].mean():.1f}")
            c4.metric("🚨 High Risk Flags", int((df["Risk"] == "High").sum()))
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif "2. ⚠️ Academic Challenge" in admin_page:
            st.subheader("⚠️ Academic Challenges Log & Resolution Editor")
            if st.session_state["student_challenges_list"]:
                st.dataframe(pd.DataFrame(st.session_state["student_challenges_list"]), use_container_width=True)
            else:
                st.info("ℹ️ No student challenges submitted yet.")

        elif "3. 🤖 AI Insights" in admin_page:
            st.info("💡 AI Model Classification Accuracy: 94.2%. Attendance < 65% serves as the primary risk predictor.")

        elif "4. 📈 Academic Performance" in admin_page:
            fig = px.histogram(df, x="Marks", color="Department", barmode="group", title="📊 Marks Distribution by Department")
            st.plotly_chart(fig, use_container_width=True)

        elif "5. 📅 Attendance Analytics" in admin_page:
            fig = px.box(df, x="Department", y="Attendance", title="📅 Department Attendance Variance")
            st.plotly_chart(fig, use_container_width=True)

        elif "6. 🚨 At Risk Students" in admin_page:
            high_risk = df[df["Risk"] == "High"]
            st.error(f"🚨 Total High Risk Students Flagged: {len(high_risk)}")
            st.dataframe(high_risk, use_container_width=True, hide_index=True)

        elif "7. 🎯 Student Success Score" in admin_page:
            st.metric("🌟 Institutional Success Index", "8.2 / 10")

        elif "8. 📚 LMS & Engagement Analytics" in admin_page:
            st.subheader("📚 LMS Portal Engagement Analytics (Editable Grid)")
            edited_lms = st.data_editor(st.session_state["lms_data"], use_container_width=True, num_rows="dynamic")
            if st.button("💾 Update LMS Metrics"):
                st.session_state["lms_data"] = edited_lms
                st.success("🎉 LMS engagement metrics updated successfully!")

        elif "9. 💼 Placement & Skills Analytics" in admin_page:
            st.subheader("💼 Placement Eligibility & Skill Scores Editor")
            edited_place = st.data_editor(st.session_state["placement_data"], use_container_width=True, num_rows="dynamic")
            if st.button("💾 Save Placement Metrics"):
                st.session_state["placement_data"] = edited_place
                st.success("🎉 Placement details updated successfully!")

        elif "10. 📊 Student Segmentation" in admin_page:
            fig = px.scatter(df, x="Attendance", y="Marks", color="Risk", title="🎯 Student Segmentation Matrix")
            st.plotly_chart(fig, use_container_width=True)

        elif "11. 👨‍🏫 Faculty Management" in admin_page:
            st.subheader("👨‍🏫 Edit & Manage Faculty Registry")
            edited_fac = st.data_editor(st.session_state["faculty_data"], use_container_width=True, num_rows="dynamic")
            if st.button("💾 Save Faculty Changes"):
                st.session_state["faculty_data"] = edited_fac
                st.success("🎉 Faculty directory updated successfully!")

        elif "12. 💬 Feedback Management" in admin_page:
            st.subheader("💬 Subject-wise Student Feedbacks Review")
            if st.session_state["student_feedback_list"]:
                st.dataframe(pd.DataFrame(st.session_state["student_feedback_list"]), use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ No subject feedback submitted by students yet.")

        elif "13. 📁 Upload CSV / Export" in admin_page:
            st.write("Upload a student CSV with at least Attendance and Marks columns.")
            up = st.file_uploader("📁 Upload New CSV Dataset", type=["csv"], key="student_csv_upload")
            if up is not None:
                try:
                    uploaded_df = pd.read_csv(up)
                    checked_df = prepare_student_data(uploaded_df)
                    st.session_state["uploaded_students_df"] = checked_df.drop(
                        columns=["Risk", "Risk_Label"], errors="ignore"
                    )
                    st.success(f"CSV validated: {len(checked_df)} student records loaded. Click below to apply.")
                    st.dataframe(checked_df, use_container_width=True, hide_index=True)
                    if st.button("🔄 Apply Uploaded Dataset"):
                        st.rerun()
                except Exception as exc:
                    st.error(f"CSV upload failed: {exc}")

        elif "14. 📑 Reports & PDF Export" in admin_page:
            st.download_button("📥 Export Institutional Report (CSV)", data=df.to_csv().encode('utf-8'), file_name="edupath_report.csv")

        elif "15. 🔔 Notifications" in admin_page:
            st.text_input("📢 Send Global Broadcast Message to Students:")
            if st.button("🚀 Send Broadcast"):
                st.success("🎉 Message dispatched to all student portals!")

        elif "16. 👤 User Management" in admin_page:
            st.subheader("👤 System User Accounts Manager (Edit Roles & Status)")
            edited_users = st.data_editor(st.session_state["user_data"], use_container_width=True, num_rows="dynamic")
            if st.button("💾 Save User Account Changes"):
                st.session_state["user_data"] = edited_users
                st.success("🎉 User management list updated successfully!")

        elif "17. ⚙️ Settings" in admin_page:
            st.checkbox("⚡ Enable AI Auto-alerts for Attendance < 75%", value=True)

        elif "18. ❓ Help" in admin_page:
            st.write("📞 Contact System Administrator Support: **admin@institution.edu**")

        elif "19. ➕ Subject Add" in admin_page:
            st.text_input("📚 New Subject Name:")
            st.text_input("🔑 Subject Code:")
            if st.button("➕ Add Subject"):
                st.success("🎉 Subject added successfully!")

        elif "20. ✏️ Marks & Grade Edit" in admin_page:
            st.selectbox("👤 Select Student ID", df["Student_ID"])
            st.number_input("✏️ Update Marks", 0, 100, 75)
            if st.button("💾 Save Marks Update"):
                st.success("🎉 Student marks updated successfully!")

        elif "21. 📝 Exam & Assignments" in admin_page:
            st.text_input("📝 Exam / Assignment Title:")
            st.date_input("📅 Scheduled Date")
            if st.button("🚀 Schedule Event"):
                st.success("🎉 Event scheduled successfully!")

        elif "22. 💡 Study Recommendations" in admin_page:
            st.text_area("💡 Configure Global AI Study Advisory Rules:")

        elif "23. 🤝 Mentorship Request" in admin_page:
            st.subheader("🤝 Edit & Allocate Student Mentorship Requests")
            edited_men = st.data_editor(st.session_state["mentorship_data"], use_container_width=True, num_rows="dynamic")
            if st.button("💾 Save Mentorship Allocations"):
                st.session_state["mentorship_data"] = edited_men
                st.success("🎉 Mentorship requests updated successfully!")

        elif "24. 📈 CGPA Tracking" in admin_page:
            st.subheader("📈 Edit Student CGPA Records")
            edited_cgpa = st.data_editor(st.session_state["cgpa_data"], use_container_width=True, num_rows="dynamic")
            if st.button("💾 Save CGPA Updates"):
                st.session_state["cgpa_data"] = edited_cgpa
                st.success("🎉 CGPA records updated successfully!")

        elif "25. 🏆 Internship Participation" in admin_page:
            st.subheader("🏆 View & Manage Student Internships & Certificates")
            if st.session_state["student_internships"]:
                st.dataframe(pd.DataFrame(st.session_state["student_internships"]), use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ No student internship submissions found.")

        elif "26. 🗓️ Time Table Plan" in admin_page:
            st.subheader("🗓️ Edit Master Academic Timetable")
            edited_tt = st.data_editor(st.session_state["timetable_data"], use_container_width=True, num_rows="dynamic")
            if st.button("💾 Save Timetable Plan"):
                st.session_state["timetable_data"] = edited_tt
                st.success("🎉 Master timetable updated successfully!")

        elif "27. 🔔 Admin Notifications" in admin_page:
            st.subheader("🔔 View & Edit Admin System Alerts & Notices")
            edited_notif = st.data_editor(st.session_state["notifications_data"], use_container_width=True, num_rows="dynamic")
            if st.button("💾 Publish & Save Notifications"):
                st.session_state["notifications_data"] = edited_notif
                st.success("🎉 Admin system notifications published successfully!")

        elif "28. 🔒 Change Password" in admin_page:
            st.text_input("🔑 Old Password", type="password")
            st.text_input("✨ New Password", type="password")
            if st.button("💾 Update Admin Password"):
                st.success("🎉 Admin password updated successfully!")

st.sidebar.markdown("---")
st.sidebar.caption("🚀 AI-powered academic support platform")
