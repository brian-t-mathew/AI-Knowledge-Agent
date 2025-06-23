import streamlit as st
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import uuid
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"

if "access_token" not in st.session_state:
    st.session_state.access_token = None
    st.session_state.user_email = None

def login(email, password):
    try:
        response = requests.post(f"{API_BASE}/login", json={"email": email, "password": password})
        if response.status_code == 200:
            data = response.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.user_email = email
            st.success("Logged in successfully!")
            st.rerun()  # Force page refresh after login
        else:
            data = response.json()
            if isinstance(data.get("detail"), list):
                messages = [item.get("msg", str(item)) for item in data["detail"]]
                st.error("Login failed: \n\n" + "\n\n".join(messages))
            else:
                st.error("Login failed: \n\n" + data.get("detail", "Unknown error"))
    except Exception as e:
        st.error(f"Login error: {str(e)}")

def register(email, password):
    try:
        response = requests.post(f"{API_BASE}/register", json={"email": email, "password": password})
        if response.status_code == 200:
            st.success("Registered successfully! Please log in.")
        else:
            data = response.json()
            if isinstance(data.get("detail"), list):
                messages = [item.get("msg", str(item)) for item in data["detail"]]
                st.error("Registration failed: \n\n" + "\n\n".join(messages))
            else:
                st.error("Registration failed: \n\n" + data.get("detail", "Unknown error"))
    except Exception as e:
        st.error(f"Registration error: {str(e)}")

def get_auth_headers():
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}

def logout():
    st.session_state.access_token = None
    st.session_state.user_email = None
    st.session_state.selected_session = None
    if "messages" in st.session_state:
        del st.session_state["messages"]
    st.rerun()

# LOGIN/LOGOUT UI in sidebar
st.sidebar.title("User Authentication")

if not st.session_state.access_token:
    tab1, tab2 = st.sidebar.tabs(["Login", "Register"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            login(email, password)

    with tab2:
        email_r = st.text_input("Email (Register)", key="reg_email")
        password_r = st.text_input("Password (Register)", type="password", key="reg_password")
        if st.button("Register"):
            register(email_r, password_r)

else:
    st.sidebar.markdown(f"Logged in as **{st.session_state.user_email}**")
    if st.sidebar.button("Logout"):
        logout()

# --- Sessions Management ---

if st.session_state.access_token:

    st.sidebar.title("Sessions")

    def fetch_sessions():
        try:
            # Remove user_id parameter - backend gets it from token
            response = requests.get(f"{API_BASE}/sessions", headers=get_auth_headers())
            if response.status_code == 200:
                return response.json().get("sessions", [])
            elif response.status_code == 401:
                st.sidebar.error("Session expired. Please log in again.")
                logout()
                return []
            else:
                st.sidebar.error(f"Failed to fetch sessions: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Failed to fetch sessions: {str(e)}")
        return []

    def create_new_session():
        try:
            response = requests.post(f"{API_BASE}/session/create", headers=get_auth_headers())
            if response.status_code == 200:
                return response.json().get("session_id")
            elif response.status_code == 401:
                st.sidebar.error("Session expired. Please log in again.")
                logout()
                return None
            else:
                st.sidebar.error(f"Failed to create session: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Failed to create session: {str(e)}")
        return str(uuid.uuid4())

    # Initialize session if not exists
    if "selected_session" not in st.session_state or st.session_state.selected_session is None:
        sessions = fetch_sessions()
        if sessions:
            st.session_state.selected_session = sessions[0]["session_id"]
            st.session_state.session_timestamp = sessions[0].get("timestamp", datetime.now().isoformat())
        else:
            new_session = create_new_session()
            if new_session:
                st.session_state.selected_session = new_session
                sessions = fetch_sessions()
                if sessions:
                    for s in sessions:
                        if s["session_id"] == st.session_state.selected_session:
                            st.session_state.session_timestamp = s.get("timestamp")
                            break
    def session_button(session):
        timestamp = session.get("timestamp")
        try:
            dt = datetime.fromisoformat(timestamp)
            local_dt = dt.astimezone()
            formatted_time = local_dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = "Unknown time"

        label = f"{session['session_id'][:8]}... ({formatted_time})"

        if st.sidebar.button(label, key=f"session_{session['session_id']}"):
            st.session_state.selected_session = session["session_id"]
            st.session_state.session_timestamp = session.get("timestamp")
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()

    st.sidebar.markdown("### Select Session")
    sessions = fetch_sessions()

    if st.sidebar.button("➕ New Session"):
        new_session = create_new_session()
        if new_session:
            st.session_state.selected_session = new_session
            sessions = fetch_sessions()
            if sessions:
                for s in sessions:
                    if s["session_id"] == st.session_state.selected_session:
                        st.session_state.session_timestamp = s.get("timestamp")
                        break
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()

    if sessions:
        for s in sessions:
            session_button(s)
    else:
        # Changed from st.sidebar.info to simple text - no more warning/error for empty sessions
        st.sidebar.markdown("*No sessions found*")

    st.sidebar.markdown("---")
    if st.session_state.selected_session:
        st.sidebar.markdown(f"**Current session:** `{st.session_state.selected_session[:12]}...`")
        if st.session_state.get("session_timestamp"):
            try:
                dt = datetime.fromisoformat(st.session_state.session_timestamp)
                local_dt = dt.astimezone()
                st.sidebar.markdown(f"**Last updated:** `{local_dt.strftime('%Y-%m-%d %H:%M:%S')}`")
            except:
                st.sidebar.markdown(f"**Last updated:** Unknown")

# --- Main Menu ---

if st.session_state.access_token:
    page = st.sidebar.selectbox("Main Menu", ["Home", "Upload", "Chatbot"], index=0)
else:
    page = "Home"  # Force home page when not logged in
    st.info("Please log in to access all features.")

if page == "Home":
    st.title("🏠 Home")
    st.markdown("### Document Chat Application")
    st.write("This app allows you to:")
    st.write("- Upload PDFs, images, or URLs")
    st.write("- Chat with an AI about the content")
    st.write("- Manage multiple conversation sessions")
    
    if not st.session_state.access_token:
        st.markdown("---")
        st.info("👈 Please log in using the sidebar to get started!")

elif page == "Upload":
    st.title("📤 Upload Documents")
    
    if not st.session_state.selected_session:
        st.warning("Please wait while we set up your session...")
        st.stop()
    
    mode = st.selectbox("Select input type", ["PDF", "Image", "URL"], key="mode_select")
    mode_mapping = {"PDF": 1, "Image": 2, "URL": 3}
    file = None
    url = None

    if mode in ["PDF", "Image"]:
        file = st.file_uploader(f"Upload a {mode} file:", type=["pdf", "png", "jpeg", "jpg"], key="file_uploader")
    elif mode == "URL":
        url = st.text_input("Enter your URL:", key="url_input")

    if st.button("Load Data", key="load_btn"):
        if mode in ["PDF", "Image"] and file is None:
            st.warning("Please upload a file.")
        elif mode == "URL" and not url:
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Processing your data..."):
                try:
                    headers = get_auth_headers()
                    if mode in ["PDF", "Image"]:
                        m = MultipartEncoder(
                            fields={
                                "mode": str(mode_mapping[mode]),
                                "session_id": st.session_state.selected_session,
                                "file": (file.name, file.read(), file.type)
                            }
                        )
                        headers["Content-Type"] = m.content_type
                        response = requests.post(
                            f"{API_BASE}/load",
                            data=m,
                            headers=headers
                        )
                    else:
                        response = requests.post(
                            f"{API_BASE}/load",
                            data={
                                "mode": str(mode_mapping[mode]),
                                "url": url,
                                "session_id": st.session_state.selected_session
                            },
                            headers=headers
                        )

                    if response.status_code == 200:
                        st.success("✅ Data loaded successfully! Switch to Chatbot to ask questions.")
                        sessions = fetch_sessions()
                        if sessions:
                            for s in sessions:
                                if s["session_id"] == st.session_state.selected_session:
                                    st.session_state.session_timestamp = s.get("timestamp")
                                    break
                    elif response.status_code == 401:
                        st.error("Session expired. Please log in again.")
                        logout()
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

elif page == "Chatbot":
    st.title("💬 Chatbot")

    if not st.session_state.selected_session:
        st.warning("Please wait while we set up your session...")
        st.stop()

    # Load or initialize messages
    if "messages" not in st.session_state or st.session_state.selected_session != st.session_state.get("last_session", ""):
        try:
            headers = get_auth_headers()
            res = requests.get(f"{API_BASE}/history", params={"session_id": st.session_state.selected_session}, headers=headers)
            if res.status_code == 200:
                st.session_state.messages = res.json().get("history", [])
            elif res.status_code == 401:
                st.error("Session expired. Please log in again.")
                logout()
                st.stop()
            else:
                st.warning(f"Failed to load history: {res.text}")
                st.session_state.messages = []
        except Exception as e:
            st.warning(f"Failed to load : {str(e)}")
            st.session_state.messages = []

        st.session_state.last_session = st.session_state.selected_session

    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_input = st.chat_input("Type your message and press Enter", key="text_input")

    prompt = user_input


    # User input
    if prompt :
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    headers = get_auth_headers()
                    response = requests.post(
                        f"{API_BASE}/chat",
                        json={
                            "session_id": st.session_state.selected_session,
                            "question": prompt
                        },
                        headers=headers
                    )
                    if response.status_code == 200:
                        reply = response.json()["answer"]
                    elif response.status_code == 401:
                        reply = "Session expired. Please log in again."
                        logout()
                    else:
                        reply = f"Error: {response.text}"
                except Exception as e:
                    reply = f"Connection error: {str(e)}"
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        sessions = fetch_sessions()
        if sessions:
            for s in sessions:
                if s["session_id"] == st.session_state.selected_session:
                    st.session_state.session_timestamp = s.get("timestamp")
                    break
        
            