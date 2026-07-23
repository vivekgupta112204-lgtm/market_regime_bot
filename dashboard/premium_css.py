"""Premium Styling configuration for Streamlit."""

def get_premium_css():
    return """
    <style>
        /* Bloomberg Terminal / Dark Mode Aesthetic */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif !important;
        }

        .stApp {
            background-color: #0B0E14;
            background-image: 
                radial-gradient(at 0% 0%, rgba(20, 20, 20, 1) 0, transparent 50%), 
                radial-gradient(at 50% 0%, rgba(30, 41, 59, 0.3) 0, transparent 50%), 
                radial-gradient(at 100% 0%, rgba(15, 23, 42, 0.4) 0, transparent 50%);
            color: #E2E8F0;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.8) !important;
            border-right: 1px solid rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
        }

        /* Metric Cards Glassmorphism */
        div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: 12px;
            padding: 15px 20px;
            transition: all 0.3s ease;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border: 1px solid rgba(56, 189, 248, 0.5);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(56, 189, 248, 0.2);
        }

        /* Value Colors - Bright Gradient */
        div[data-testid="stMetricValue"] {
            font-weight: 800 !important;
            font-size: 2rem !important;
            background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Headers */
        h1, h2, h3 {
            color: #F8FAFC !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em;
        }
        
        /* Dataframes styling */
        .stDataFrame {
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        
        /* Badges/Tags Customization inside markdown */
        .stMarkdown p {
            color: #CBD5E1;
        }
    </style>
    """
