import os
import json
import streamlit as st
from datetime import datetime
from src.graph.state import LoanApplicationState, HumanDecision, ExtractedDocument
from src.graph.graph import graph

st.set_page_config(
    page_title="Loan Underwriter Officer Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background-color: #0f172a;
    }
    h1, h2, h3, p, span, li {
        color: #f8fafc !important;
    }
    .reportview-container {
        background: #0f172a;
    }
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .agent-header {
        color: #38bdf8 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load Synthetic Data
TEST_DATA_PATH = "./data/test_applications.json"

@st.cache_data
def load_applications():
    if os.path.exists(TEST_DATA_PATH):
        with open(TEST_DATA_PATH, "r") as f:
            return json.load(f)
    return []

applications = load_applications()

st.sidebar.title("🏦 Underwriter Control Panel")
st.sidebar.write("Select a synthetic application to test the multi-agent system:")

if not applications:
    st.sidebar.error("No test applications found. Please run indexing first.")
    selected_app = None
else:
    app_options = {f"{app['application_id']} - {app['applicant_data']['name']}": app for app in applications}
    selected_option = st.sidebar.selectbox("Choose Application:", list(app_options.keys()))
    selected_app = app_options[selected_option]

st.title("🏦 Loan Approval Multi-Agent Underwriter")
st.subheader("Officer Verification Dashboard (Human-in-the-Loop)")

if selected_app:
    # Display selected applicant data
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.write("**Applicant Name**")
        st.write(selected_app["applicant_data"]["name"])
        st.write(f"**Annual Income:** ${selected_app['applicant_data']['income']:,}")
        st.write(f"**Monthly Debt:** ${selected_app['applicant_data']['monthly_debt']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.write("**Loan Details**")
        st.write(f"**Loan Amount Requested:** ${selected_app['applicant_data']['loan_amount']:,}")
        st.write(f"**Collateral / Property Value:** ${selected_app['applicant_data']['property_value']:,}")
        ltv = selected_app['applicant_data']['loan_amount'] / selected_app['applicant_data']['property_value']
        st.write(f"**Estimated LTV:** {ltv:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.write("**Employment & Target**")
        st.write(f"**Employment History:** {selected_app['applicant_data']['employment_months']} months")
        st.write(f"**Expected Outcome:** :orange[{selected_app.get('expected_recommendation', 'N/A').upper()}]")
        st.write(f"**Scenario:** *{selected_app.get('description', '')}*")
        st.markdown('</div>', unsafe_allow_html=True)

    # Initialize session state for graph execution
    if "graph_state" not in st.session_state or st.session_state.get("current_app_id") != selected_app["application_id"]:
        st.session_state["graph_state"] = None
        st.session_state["current_app_id"] = selected_app["application_id"]
        st.session_state["officer_action_submitted"] = False

    if st.button("⚡ Run Multi-Agent Underwriting pipeline"):
        with st.spinner("Orchestrating agents via LangGraph..."):
            # Prepare state
            docs_list = []
            for doc in selected_app.get("documents", []):
                docs_list.append(ExtractedDocument(
                    document_type=doc["document_type"],
                    extracted_text=doc["extracted_text"],
                    validation_status=doc["validation_status"],
                    confidence=doc["confidence"],
                    extracted_fields=doc["extracted_fields"]
                ))
                
            initial_state = LoanApplicationState(
                application_id=selected_app["application_id"],
                applicant_data=selected_app["applicant_data"],
                documents=docs_list
            )
            
            # Execute LangGraph up to the human_review node
            # In our setup, the graph will run through kyc -> credit -> policy -> arbitrator -> human_review
            final_state_dict = graph.invoke(initial_state)
            st.session_state["graph_state"] = final_state_dict
            st.session_state["officer_action_submitted"] = False
            st.success("Graph execution completed successfully!")

    # Display results if graph has run
    state_data = st.session_state["graph_state"]
    if state_data:
        # Load typed Pydantic models from execution outputs
        kyc_out = state_data.get("kyc_output")
        credit_out = state_data.get("credit_output")
        policy_out = state_data.get("policy_output")
        arb_out = state_data.get("arbitrator_output")
        
        st.divider()
        st.header("🤖 Multi-Agent Assessment Results")
        
        # 1. Arbitrator Final Recommendation
        if arb_out:
            rec_color = {
                "approve": "green",
                "deny": "red",
                "review_required": "orange"
            }.get(arb_out.recommendation, "blue")
            
            st.markdown(f"""
            <div style="background-color: #1e293b; padding: 1.5rem; border-radius: 12px; border: 2px solid #3b82f6; margin-bottom: 2rem;">
                <h3 style="margin-top:0;">Arbitrator Recommendation: <span style="color: {rec_color}; font-weight: bold;">{arb_out.recommendation.upper()}</span></h3>
                <p><b>Confidence Score:</b> {arb_out.confidence_score:.2f} | <b>Agreement:</b> {arb_out.agent_agreement.upper()}</p>
                <p><b>Summary:</b> {arb_out.summary}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # 2. Detailed Agent Outputs side-by-side
        col_kyc, col_credit, col_policy = st.columns(3)
        
        with col_kyc:
            st.markdown('<h4 class="agent-header">📄 KYC Agent</h4>', unsafe_allow_html=True)
            if kyc_out:
                st.write(f"**Verification status:** {'Passed' if not kyc_out.get('missing_critical_docs') else 'Failed'}")
                st.write(f"**Fraud/Suspicion Flag:** {'⚠️ Yes' if kyc_out.get('fraud_flag') else 'No'}")
                st.write(f"**Confidence:** {kyc_out.get('confidence', 1.0):.2f}")
                st.json(kyc_out.get("verified_fields", {}))
            else:
                st.info("No KYC output available.")
                
        with col_credit:
            st.markdown('<h4 class="agent-header">💳 Credit Agent</h4>', unsafe_allow_html=True)
            if credit_out:
                st.write(f"**Calculated Score:** `{credit_out.credit_score}`")
                st.write(f"**Risk Level:** :orange[{credit_out.risk_category.upper()}]")
                st.write(f"**DTI Ratio:** {credit_out.dti_ratio:.2%}")
                st.write(f"**Reasoning:** *{credit_out.reasoning}*")
            else:
                st.info("No credit assessment output available.")
                
        with col_policy:
            st.markdown('<h4 class="agent-header">📋 Policy Agent</h4>', unsafe_allow_html=True)
            if policy_out:
                st.write(f"**Policy Passed:** {'Yes' if policy_out.policy_passed else '❌ No'}")
                st.write(f"**LTV Ratio:** {policy_out.ltv_ratio:.2%}")
                if policy_out.violations:
                    st.write("**Violations detected:**")
                    for v in policy_out.violations:
                        st.write(f"- :red[{v}]")
            else:
                st.info("No policy verification output available.")

        # RAG grounded policy evidence
        if policy_out and policy_out.retrieved_policy_chunks:
            with st.expander("📖 Grounded Policy Evidence (RAG Retrieved Chunks)"):
                for idx, chunk in enumerate(policy_out.retrieved_policy_chunks):
                    st.markdown(f"**Chunk #{idx+1}**")
                    st.info(chunk)

        # 3. Human-in-the-Loop Override Section
        st.divider()
        st.subheader("⚖️ Human-in-the-Loop Decision")
        
        if not st.session_state["officer_action_submitted"]:
            with st.form("hitl_review_form"):
                st.write("Review the agent reports and make a final binding underwriter decision:")
                
                default_rec = arb_out.recommendation if arb_out else "review_required"
                
                decision_mapping = {
                    "Approve (Agree with Agent)": "approve",
                    "Deny (Agree with Agent)": "deny",
                    "OVERRIDE: Approve Application": "override_approve",
                    "OVERRIDE: Deny Application": "override_deny"
                }
                
                selected_decision_label = st.radio("Final Decision:", list(decision_mapping.keys()))
                decision = decision_mapping[selected_decision_label]
                
                override_reason = st.text_area("Reasoning / Override Justification:", 
                                               help="Mandatory if overriding agent's recommendation or for borderline cases.")
                
                officer_id = st.text_input("Officer ID / Signature:", value="OFFICER-2026-009")
                
                submit = st.form_submit_state = st.form_submit_button("Submit Binding Underwriter Decision")
                
                if submit:
                    if ("override" in decision or default_rec != decision) and not override_reason.strip():
                        st.error("Override reasoning is mandatory when overriding agent recommendation.")
                    elif not officer_id.strip():
                        st.error("Officer ID is required to log the audit trail.")
                    else:
                        # Construct human decision state payload
                        human_decision = HumanDecision(
                            officer_id=officer_id,
                            decision=decision,
                            override_reason=override_reason if override_reason.strip() else None,
                            timestamp=datetime.now().isoformat()
                        )
                        
                        # Re-run or update state in graph to execute the human review node
                        final_state_data = dict(state_data)
                        final_state_data["human_decision"] = human_decision
                        
                        # Run graph with human decision supplied
                        updated_state = graph.invoke(final_state_data)
                        st.session_state["graph_state"] = updated_state
                        st.session_state["officer_action_submitted"] = True
                        st.rerun()
        else:
            # Display decision audit log
            human_dec = state_data.get("human_decision")
            final_status = state_data.get("final_status")
            
            st.success(f"### Decision Recorded: {final_status.upper()}")
            st.write(f"**Reviewing Officer:** {human_dec.officer_id}")
            st.write(f"**Decision Type:** {human_dec.decision.upper()}")
            if human_dec.override_reason:
                st.write(f"**Justification:** *{human_dec.override_reason}*")
            st.write(f"**Timestamp:** {human_dec.timestamp}")
            
            if st.button("🔄 Review Another Application"):
                st.session_state["graph_state"] = None
                st.session_state["officer_action_submitted"] = False
                st.rerun()
else:
    st.info("Please select an application from the sidebar to begin underwriting reviews.")
