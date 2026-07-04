import streamlit as st
import os
import json
import chromadb
import anthropic
from datetime import datetime
import uuid
from jinja2 import Template
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# ── Page config ──
st.set_page_config(
    page_title="TradeBot — Textile Export Agent",
    page_icon="🚢",
    layout="wide"
)

# ── Paths ── AZURE COMPATIBLE
base = os.path.dirname(os.path.abspath(__file__))
kb_path = os.path.join(base, "knowledge_base")
outputs_path = os.path.join(os.getcwd(), "outputs")
os.makedirs(outputs_path, exist_ok=True)

# ── API Key ── AZURE COMPATIBLE (set in Azure App Service > Configuration)
ANTHROPIC_API_KEY = "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"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INITIALISE — cached so it runs once only
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def initialise():
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="tradebot_app_kb",
        metadata={"hnsw:space": "cosine"}
    )
    if collection.count() == 0:
        kb_files = [
            "destination_country_requirements.txt",
            "hs_codes_textile.txt",
            "incoterms_guide.txt",
            "lc_requirements.txt",
            "pakistan_export_regulations.txt",
            "trade_agreements.txt",
            "certification_guide.txt",
        ]
        def chunk_text(text, chunk_size=500, overlap=50):
            words = text.split()
            chunks = []
            start = 0
            while start < len(words):
                chunks.append(" ".join(words[start:start+chunk_size]))
                start += chunk_size - overlap
            return chunks

        for kb_file in kb_files:
            with open(os.path.join(kb_path, kb_file), "r", encoding="utf-8") as f:
                text = f.read()
            chunks = chunk_text(text)
            source_name = kb_file.replace(".txt", "")
            collection.add(
                documents=chunks,
                metadatas=[{"source": source_name, "chunk_index": i} for i, _ in enumerate(chunks)],
                ids=[f"app_{source_name}_chunk_{i}" for i, _ in enumerate(chunks)]
            )
    return client, collection


client, collection = initialise()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def retrieve_knowledge(query, n_results=3, source_filter=None):
    where_clause = {"source": source_filter} if source_filter else None
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause
    )
    retrieved = []
    for i, doc in enumerate(results["documents"][0]):
        retrieved.append({
            "content": doc,
            "source": results["metadatas"][0][i]["source"],
        })
    return retrieved

def format_retrieved(results):
    context = ""
    for r in results:
        context += f"\n[SOURCE: {r['source']}]\n{r['content']}\n"
    return context

def claude_call(prompt, max_tokens=1500):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT NODES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TradeBotState(TypedDict):
    exporter_input: dict
    validated_input: dict
    input_errors: List[str]
    product_classification: dict
    country_requirements: str
    trade_agreement_info: str
    document_plan: dict
    commercial_invoice_data: dict
    packing_list_data: dict
    hs_code_report: dict
    compliance_check: dict
    readiness_score: dict
    final_package: dict
    output_folder: str


def node_input_collector(state):
    data = state["exporter_input"]
    errors = []
    required = ["company", "product", "composition", "destination",
                "buyer", "fob_value_usd", "incoterm", "payment"]
    for field in required:
        if field not in data or not data[field]:
            errors.append(f"Missing: {field}")
    validated = data.copy()
    validated["incoterm"] = data.get("incoterm", "FOB").upper()
    validated["payment"] = data.get("payment", "TT").upper()
    validated["certifications"] = data.get("certifications", [])
    quantity = validated.get("quantity_meters", validated.get("quantity_pieces", 0))
    validated["quantity"] = quantity
    validated["unit"] = "Metres" if validated.get("quantity_meters", 0) > 0 else "Pieces"
    return {"validated_input": validated, "input_errors": errors}


def node_product_classifier(state):
    data = state["validated_input"]
    hs_context = format_retrieved(retrieve_knowledge(
        f"HS code {data['product']} {data['composition']}",
        n_results=2, source_filter="hs_codes_textile"
    ))
    prompt = f"""Classify this textile product and suggest HS code.
Return ONLY valid JSON — no markdown, no backticks.
PRODUCT: {data['product']}
COMPOSITION: {data['composition']}
GSM: {data.get('gsm', 'unknown')}
KNOWLEDGE: {hs_context}
Return: {{"product_category":"...","hs_code_suggested":"...","hs_description":"...","confidence":"High/Medium/Low","classification_notes":"..."}}"""
    result = claude_call(prompt, max_tokens=500)
    return {"product_classification": result}


def node_country_retriever(state):
    data = state["validated_input"]
    country_results = retrieve_knowledge(
        f"export requirements documents {data['destination']} textile",
        n_results=3, source_filter="destination_country_requirements"
    )
    trade_results = retrieve_knowledge(
        f"Pakistan GSP trade agreement {data['destination']} duty",
        n_results=2, source_filter="trade_agreements"
    )
    return {
        "country_requirements": format_retrieved(country_results),
        "trade_agreement_info": format_retrieved(trade_results)
    }


def node_document_planner(state):
    data = state["validated_input"]
    classification = state["product_classification"]
    country_req = state["country_requirements"]
    prompt = f"""You are a textile export documentation planner.
Return ONLY valid JSON — no markdown, no backticks.
DESTINATION: {data['destination']} | INCOTERM: {data['incoterm']} | PAYMENT: {data['payment']}
CATEGORY: {classification['product_category']} | CERTS: {data['certifications']}
COUNTRY REQUIREMENTS: {country_req[:500]}
Return: {{"documents_required":[],"documents_to_generate":[],"certificate_of_origin_type":"...","insurance_required":true,"lc_checklist_required":true,"special_requirements":[],"missing_exporter_inputs":[]}}"""
    result = claude_call(prompt, max_tokens=800)
    return {"document_plan": result}


def node_document_generator(state):
    data = state["validated_input"]
    classification = state["product_classification"]
    ci_prompt = f"""Generate commercial invoice data for textile export.
Return ONLY valid JSON — no markdown, no backticks.
EXPORTER: {json.dumps(data)} | HS CODE: {classification['hs_code_suggested']}
Return: {{"invoice_number":"INV-{datetime.now().strftime('%Y%m%d')}-XXXX","invoice_date":"{datetime.now().strftime('%d-%b-%Y')}","lc_number":"N/A","exporter_company":"{data['company']}","buyer_company":"{data['buyer']}","buyer_address":"buyer address","destination_country":"{data['destination']}","port_of_loading":"Port Qasim, Karachi, Pakistan","port_of_discharge":"main port of {data['destination']}","incoterm":"{data['incoterm']}","payment_terms":"{data['payment']}","hs_code":"{classification['hs_code_suggested']}","product_description":"formal description","composition":"{data['composition']}","gsm":"{data.get('gsm','N/A')}","quantity":{data['quantity']},"unit":"{data['unit']}","unit_price":"calculated","total_value":{data['fob_value_usd']},"certifications":{json.dumps(data['certifications'])},"ntn":"XXXXXXXX","strn":"XXXXXXXX"}}"""
    ci_data = claude_call(ci_prompt, max_tokens=1000)

    pl_prompt = f"""Generate packing list data for textile export.
Return ONLY valid JSON — no markdown, no backticks.
EXPORTER: {json.dumps(data)} | INVOICE: {ci_data['invoice_number']}
Return: {{"packing_list_number":"PL-{datetime.now().strftime('%Y%m%d')}-XXXX","date":"{datetime.now().strftime('%d-%b-%Y')}","invoice_number":"{ci_data['invoice_number']}","exporter_company":"{data['company']}","buyer_company":"{data['buyer']}","destination_country":"{data['destination']}","port_of_loading":"Port Qasim, Karachi, Pakistan","port_of_discharge":"main port of {data['destination']}","incoterm":"{data['incoterm']}","product_description":"formal description","hs_code":"{classification['hs_code_suggested']}","quantity":{data['quantity']},"unit":"{data['unit']}","num_cartons":"realistic number","qty_per_carton":"realistic","carton_length":"cm","carton_width":"cm","carton_height":"cm","net_weight_per_carton":"kg","gross_weight_per_carton":"kg","total_net_weight":"calculated","total_gross_weight":"calculated","marks_and_numbers":"{data['company']} / {data['buyer']} / MADE IN PAKISTAN"}}"""
    pl_data = claude_call(pl_prompt, max_tokens=1000)
    return {"commercial_invoice_data": ci_data, "packing_list_data": pl_data}


def node_hs_validator(state):
    data = state["validated_input"]
    classification = state["product_classification"]
    trade_info = state["trade_agreement_info"]
    prompt = f"""Validate HS code and calculate duty savings.
Return ONLY valid JSON — no markdown, no backticks.
PRODUCT: {data['product']} | COMPOSITION: {data['composition']}
HS CODE: {classification['hs_code_suggested']} | DESTINATION: {data['destination']}
FOB VALUE: {data['fob_value_usd']} | TRADE INFO: {trade_info[:500]}
Return: {{"product":"{data['product']}","composition":"{data['composition']}","gsm":"{data.get('gsm','N/A')}","hs_code_6digit":"...","hs_code_destination":"...","hs_description":"...","destination_country":"{data['destination']}","mfn_duty_rate":"...","gsp_applicable":true,"gsp_scheme":"...","gsp_duty_rate":"...","duty_saving_usd":"...","trade_agreement_notes":"...","origin_criterion":"P or W","recommendations":[]}}"""
    result = claude_call(prompt, max_tokens=1000)
    return {"hs_code_report": result}


def node_compliance_checker(state):
    data = state["validated_input"]
    doc_plan = state["document_plan"]
    hs_data = state["hs_code_report"]
    lc_context = format_retrieved(retrieve_knowledge(
        "LC discrepancies textile exports UCP 600",
        n_results=2, source_filter="lc_requirements"
    ))
    cert_context = format_retrieved(retrieve_knowledge(
        f"certification requirements {data['destination']} buyer",
        n_results=2, source_filter="certification_guide"
    ))
    prompt = f"""Check textile export compliance risks.
Return ONLY valid JSON — no markdown, no backticks.
COMPANY: {data['company']} | DESTINATION: {data['destination']}
INCOTERM: {data['incoterm']} | PAYMENT: {data['payment']}
CERTS: {data['certifications']} | HS CODE: {hs_data['hs_code_6digit']}
LC KNOWLEDGE: {lc_context[:400]}
CERT KNOWLEDGE: {cert_context[:400]}
Return: {{"compliance_status":"Pass/Partial/Fail","lc_risk_level":"Low/Medium/High/Not Applicable","lc_risks":[],"certification_gaps":[],"certification_ok":[],"document_gaps":[],"regulatory_flags":[],"compliance_notes":"..."}}"""
    result = claude_call(prompt, max_tokens=1200)
    return {"compliance_check": result}


def node_readiness_scorer(state):
    data = state["validated_input"]
    doc_plan = state["document_plan"]
    hs_data = state["hs_code_report"]
    compliance = state["compliance_check"]
    prompt = f"""Score this textile export package 0-100.
Return ONLY valid JSON — no markdown, no backticks.
COMPANY: {data['company']} | PRODUCT: {data['product']}
DESTINATION: {data['destination']} | INCOTERM: {data['incoterm']}
PAYMENT: {data['payment']} | CERTS: {data['certifications']}
FOB VALUE: USD {data['fob_value_usd']}
COMPLIANCE: {json.dumps(compliance)}
HS CODE: {hs_data['hs_code_6digit']} | GSP: {hs_data['gsp_applicable']}
SCORING: 90-100=Ready | 75-89=Minor gaps | 60-74=Moderate | 40-59=Significant | <40=Critical
Return: {{"score":75,"grade":"B","summary":"one sentence","complete_items":[],"missing_items":[],"risk_flags":[],"action_items":[],"lc_risk_level":"{compliance['lc_risk_level']}","gsp_opportunity":"..."}}"""
    result = claude_call(prompt, max_tokens=1200)
    return {"readiness_score": result}


def node_final_synthesizer(state):
    data = state["validated_input"]
    ci_data = state["commercial_invoice_data"]
    pl_data = state["packing_list_data"]
    hs_data = state["hs_code_report"]
    rs_data = state["readiness_score"]
    compliance = state["compliance_check"]

    company_clean = data['company'].replace(" ", "_")
    folder_name = f"{company_clean}_{data['destination']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_folder = os.path.join(outputs_path, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    ci_template = Template("""COMMERCIAL INVOICE
==================
Invoice Number  : {{ invoice_number }}
Invoice Date    : {{ invoice_date }}
LC Number       : {{ lc_number }}

EXPORTER: {{ exporter_company }}, Karachi, Pakistan
NTN: {{ ntn }} | STRN: {{ strn }}

BUYER: {{ buyer_company }}
Address: {{ buyer_address }}
Country: {{ destination_country }}

Port of Loading    : {{ port_of_loading }}
Port of Discharge  : {{ port_of_discharge }}
Incoterm           : {{ incoterm }}
Payment Terms      : {{ payment_terms }}
Country of Origin  : Pakistan

HS Code        : {{ hs_code }}
Description    : {{ product_description }}
Composition    : {{ composition }}
GSM            : {{ gsm }}
Quantity       : {{ quantity }} {{ unit }}
Unit Price     : USD {{ unit_price }}
Total Value    : USD {{ total_value }}

Certifications : {% if certifications %}{{ certifications | join(', ') }}{% else %}None{% endif %}

Authorized Signatory: _______________________
""")

    pl_template = Template("""PACKING LIST
============
PL Number      : {{ packing_list_number }}
Date           : {{ date }}
Invoice Ref    : {{ invoice_number }}

Exporter: {{ exporter_company }}, Karachi, Pakistan
Buyer   : {{ buyer_company }}, {{ destination_country }}

Port of Loading    : {{ port_of_loading }}
Port of Discharge  : {{ port_of_discharge }}
Incoterm           : {{ incoterm }}

Product     : {{ product_description }}
HS Code     : {{ hs_code }}
Quantity    : {{ quantity }} {{ unit }}
Cartons     : {{ num_cartons }}
Per Carton  : {{ qty_per_carton }} {{ unit }}
Dimensions  : {{ carton_length }}cm x {{ carton_width }}cm x {{ carton_height }}cm
Net/Carton  : {{ net_weight_per_carton }} kg
Gross/Carton: {{ gross_weight_per_carton }} kg

Total Net Weight  : {{ total_net_weight }} kg
Total Gross Weight: {{ total_gross_weight }} kg

Marks: {{ marks_and_numbers }}

Authorized Signatory: _______________________
""")

    docs = {}
    docs["01_commercial_invoice.txt"] = ci_template.render(**ci_data)
    docs["02_packing_list.txt"] = pl_template.render(**pl_data)
    docs["03_hs_code_report.txt"] = f"""HS CODE REPORT
==============
Product          : {hs_data['product']}
HS Code (6-digit): {hs_data['hs_code_6digit']}
Destination Code : {hs_data['hs_code_destination']}
Description      : {hs_data['hs_description']}
Destination      : {hs_data['destination_country']}
MFN Duty Rate    : {hs_data['mfn_duty_rate']}
GSP Applicable   : {hs_data['gsp_applicable']}
GSP Scheme       : {hs_data.get('gsp_scheme','None')}
Duty Saving USD  : {hs_data['duty_saving_usd']}
Origin Criterion : {hs_data.get('origin_criterion','N/A')}

Trade Notes: {hs_data.get('trade_agreement_notes','None')}

Recommendations:
{chr(10).join(f'- {r}' for r in hs_data.get('recommendations',[]))}
"""
    docs["04_compliance_report.txt"] = f"""COMPLIANCE REPORT
=================
Status        : {compliance['compliance_status']}
LC Risk Level : {compliance['lc_risk_level']}

LC Risks:
{chr(10).join(f'- {r}' for r in compliance.get('lc_risks',[]))}

Certification OK   : {', '.join(compliance.get('certification_ok',['None']))}
Certification Gaps : {', '.join(compliance.get('certification_gaps',['None']))}

Document Gaps:
{chr(10).join(f'- {d}' for d in compliance.get('document_gaps',[]))}

Notes: {compliance.get('compliance_notes','')}
"""
    docs["05_export_readiness_score.txt"] = f"""EXPORT READINESS SCORE
======================
Exporter    : {data['company']}
Destination : {data['destination']}
Score       : {rs_data['score']}/100 — Grade {rs_data['grade']}
{rs_data['summary']}

Complete:
{chr(10).join(f'✅ {i}' for i in rs_data.get('complete_items',[]))}

Missing:
{chr(10).join(f'❌ {i}' for i in rs_data.get('missing_items',[]))}

Risks:
{chr(10).join(f'⚠️  {f}' for f in rs_data.get('risk_flags',[]))}

Actions:
{chr(10).join(f'{i+1}. {a}' for i,a in enumerate(rs_data.get('action_items',[])))}

GSP Opportunity: {rs_data.get('gsp_opportunity','None')}
"""
    docs["00_cover_summary.txt"] = f"""TRADEBOT — EXPORT PACKAGE
==========================
Generated  : {datetime.now().strftime('%d-%b-%Y %H:%M')}
Exporter   : {data['company']}
Product    : {data['product']}
Destination: {data['destination']}
Buyer      : {data['buyer']}
Incoterm   : {data['incoterm']}
Payment    : {data['payment']}
FOB Value  : USD {data['fob_value_usd']:,}
Score      : {rs_data['score']}/100 — Grade {rs_data['grade']}

Generated by TradeBot | Built by Junaid Iqbal | iamjunaidiqbal.com
"""
    for filename, content in docs.items():
        with open(os.path.join(output_folder, filename), "w", encoding="utf-8") as f:
            f.write(content)

    return {
        "final_package": {
            "company": data['company'],
            "destination": data['destination'],
            "score": rs_data['score'],
            "grade": rs_data['grade'],
            "docs": docs,
            "output_folder": output_folder
        },
        "output_folder": output_folder
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def build_agent():
    workflow = StateGraph(TradeBotState)
    workflow.add_node("input_collector", node_input_collector)
    workflow.add_node("product_classifier", node_product_classifier)
    workflow.add_node("country_retriever", node_country_retriever)
    workflow.add_node("document_planner", node_document_planner)
    workflow.add_node("document_generator", node_document_generator)
    workflow.add_node("hs_validator", node_hs_validator)
    workflow.add_node("compliance_checker", node_compliance_checker)
    workflow.add_node("readiness_scorer", node_readiness_scorer)
    workflow.add_node("final_synthesizer", node_final_synthesizer)
    workflow.set_entry_point("input_collector")
    workflow.add_edge("input_collector", "product_classifier")
    workflow.add_edge("product_classifier", "country_retriever")
    workflow.add_edge("country_retriever", "document_planner")
    workflow.add_edge("document_planner", "document_generator")
    workflow.add_edge("document_generator", "hs_validator")
    workflow.add_edge("hs_validator", "compliance_checker")
    workflow.add_edge("compliance_checker", "readiness_scorer")
    workflow.add_edge("readiness_scorer", "final_synthesizer")
    workflow.add_edge("final_synthesizer", END)
    return workflow.compile()


agent = build_agent()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STREAMLIT UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.title("🚢 TradeBot — Textile Export Documentation Agent")
st.caption("Generate complete export document packages in minutes | Built by Junaid Iqbal")
st.divider()

col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.subheader("📋 Exporter Details")

    company = st.text_input("Company Name", placeholder="Karachi Cotton Mills")
    product = st.text_input("Product Description", placeholder="100% Cotton Jersey Fabric")
    composition = st.text_input("Fibre Composition", placeholder="100% Cotton")
    gsm = st.number_input("GSM (0 if not applicable)", min_value=0, max_value=2000, value=180)

    col1, col2 = st.columns(2)
    with col1:
        destination = st.selectbox("Destination Country", [
            "USA", "UK", "Germany", "France", "Italy",
            "UAE", "Saudi Arabia", "Australia"
        ])
    with col2:
        incoterm = st.selectbox("Incoterm", [
            "FOB", "CIF", "CFR", "CIP", "DAP", "DDP", "EXW"
        ])

    buyer = st.text_input("Buyer Name", placeholder="Walmart Sourcing LLC")
    buyer_address = st.text_input("Buyer Address", placeholder="702 SW 8th Street, Bentonville, AR, USA")

    col3, col4 = st.columns(2)
    with col3:
        qty_meters = st.number_input("Quantity (Metres)", min_value=0, value=0)
    with col4:
        qty_pieces = st.number_input("Quantity (Pieces)", min_value=0, value=0)

    fob_value = st.number_input("FOB Value (USD)", min_value=0, value=50000)

    payment = st.selectbox("Payment Method", ["LC", "TT", "CAD", "DP"])

    st.markdown("**Certifications Available**")
    cert_col1, cert_col2 = st.columns(2)
    with cert_col1:
        has_oekotex = st.checkbox("OEKO-TEX")
        has_gots = st.checkbox("GOTS")
    with cert_col2:
        has_grs = st.checkbox("GRS")
        has_reach = st.checkbox("REACH")

    certifications = []
    if has_oekotex: certifications.append("OEKO-TEX")
    if has_gots: certifications.append("GOTS")
    if has_grs: certifications.append("GRS")
    if has_reach: certifications.append("REACH")

    generate_btn = st.button("🚀 GENERATE EXPORT PACKAGE", type="primary", use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN AGENT ON BUTTON CLICK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with col_right:
    if generate_btn:
        if not company or not product or not composition or not buyer:
            st.error("❌ Please fill in Company, Product, Composition, and Buyer fields.")
        else:
            exporter_data = {
                "company": company,
                "product": product,
                "composition": composition,
                "gsm": gsm,
                "destination": destination,
                "buyer": buyer,
                "buyer_address": buyer_address,
                "quantity_meters": qty_meters,
                "quantity_pieces": qty_pieces,
                "fob_value_usd": fob_value,
                "incoterm": incoterm,
                "payment": payment,
                "certifications": certifications
            }

            with st.spinner("🤖 TradeBot is generating your export package..."):
                progress = st.progress(0)
                status = st.empty()

                status.text("Node 1/9 — Validating inputs...")
                progress.progress(10)

                result = agent.invoke({
                    "exporter_input": exporter_data,
                    "validated_input": {},
                    "input_errors": [],
                    "product_classification": {},
                    "country_requirements": "",
                    "trade_agreement_info": "",
                    "document_plan": {},
                    "commercial_invoice_data": {},
                    "packing_list_data": {},
                    "hs_code_report": {},
                    "compliance_check": {},
                    "readiness_score": {},
                    "final_package": {},
                    "output_folder": ""
                })

                progress.progress(100)
                status.text("✅ Package complete!")

            st.success("✅ Export package generated successfully!")

            # ── Output tabs ──
            tab1, tab2, tab3 = st.tabs([
                "📄 Document Package",
                "📊 Export Readiness",
                "🔢 HS Code Report"
            ])

            with tab1:
                st.subheader("Generated Documents")
                docs = result['final_package']['docs']

                for filename, content in docs.items():
                    if filename == "00_cover_summary.txt":
                        continue
                    label = filename.replace(".txt", "").replace("_", " ").title()
                    with st.expander(f"📄 {label}", expanded=False):
                        st.text(content)
                        st.download_button(
                            label=f"⬇️ Download {label}",
                            data=content,
                            file_name=filename,
                            mime="text/plain",
                            key=filename
                        )

                all_docs = "\n\n" + "="*60 + "\n\n".join(
                    [f"FILE: {k}\n\n{v}" for k, v in docs.items()]
                )
                st.download_button(
                    label="⬇️ Download Complete Package (All Files)",
                    data=all_docs,
                    file_name=f"TradeBot_{company.replace(' ','_')}_{destination}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with tab2:
                rs = result['readiness_score']
                score = rs['score']
                grade = rs['grade']

                col_score, col_grade = st.columns(2)
                with col_score:
                    st.metric("Export Readiness Score", f"{score}/100")
                with col_grade:
                    st.metric("Grade", grade)

                st.progress(score / 100)
                st.info(rs['summary'])

                col_ok, col_missing = st.columns(2)
                with col_ok:
                    st.markdown("**✅ Complete**")
                    for item in rs.get('complete_items', []):
                        st.markdown(f"✅ {item}")

                with col_missing:
                    st.markdown("**❌ Missing**")
                    for item in rs.get('missing_items', []):
                        st.markdown(f"❌ {item}")

                if rs.get('risk_flags'):
                    st.markdown("**⚠️ Risk Flags**")
                    for flag in rs['risk_flags']:
                        st.warning(flag)

                if rs.get('action_items'):
                    st.markdown("**📋 Action Items**")
                    for i, action in enumerate(rs['action_items'], 1):
                        st.markdown(f"{i}. {action}")

                gsp = rs.get('gsp_opportunity', '')
                if gsp and gsp != 'None':
                    st.success(f"💰 GSP Opportunity: {gsp}")

            with tab3:
                hs = result['hs_code_report']

                col_hs1, col_hs2 = st.columns(2)
                with col_hs1:
                    st.metric("HS Code (6-digit)", hs['hs_code_6digit'])
                    st.metric("MFN Duty Rate", hs['mfn_duty_rate'])
                with col_hs2:
                    st.metric("GSP Applicable", "Yes ✅" if hs['gsp_applicable'] else "No ❌")
                    st.metric("Duty Saving (USD)", f"USD {hs['duty_saving_usd']}")

                st.markdown(f"**HS Description:** {hs['hs_description']}")
                st.markdown(f"**GSP Scheme:** {hs.get('gsp_scheme', 'None')}")
                st.markdown(f"**Origin Criterion:** {hs.get('origin_criterion', 'N/A')}")
                st.markdown(f"**Trade Notes:** {hs.get('trade_agreement_notes', 'None')}")

                if hs.get('recommendations'):
                    st.markdown("**Recommendations:**")
                    for rec in hs['recommendations']:
                        st.markdown(f"- {rec}")

    else:
        st.info("👈 Fill in the exporter details on the left and click **GENERATE EXPORT PACKAGE**")
        st.markdown("""
**TradeBot generates:**
- 📄 Commercial Invoice
- 📦 Packing List  
- 🔢 HS Code Report
- ✅ Compliance Report
- 📊 Export Readiness Score

**Powered by:**
- 🤖 Claude AI (Anthropic)
- 🔗 LangGraph 9-node pipeline
- 📚 RAG knowledge base
- 🌍 7 destination countries
""")