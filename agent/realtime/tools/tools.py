import asyncio
from datetime import datetime
from agent.realtime.tools.tool import tool


@tool(description="Get the current local time")
def get_current_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


@tool(
    description="Analyze complex data and generate insights", 
    long_running=True,
    loading_message="Ich analysiere gerade die Daten und generiere Insights... Das kann einen Moment dauern.",
    result_context="Die Analyse sollte als strukturierte Zusammenfassung mit Erkenntnissen präsentiert werden."
)
async def mock_long_running_analysis(query: str = "default analysis") -> str:
    """Mock tool that simulates a long-running data analysis."""
    # Simulate long-running work
    await asyncio.sleep(10)
    
    # Return semantic mock response
    return f"""📊 Analyse-Ergebnisse für: "{query}"

    🔍 **Wichtigste Erkenntnisse:**
    - Trend 1: Starke Korrelation zwischen Variablen A und B (r=0.87)
    - Trend 2: Saisonale Muster alle 3 Monate erkennbar
    - Anomalie: Ungewöhnlicher Spike am 15. des Monats

    📈 **Empfehlungen:**
    1. Fokus auf Variable A zur Optimierung
    2. Berücksichtigung saisonaler Effekte bei Planung  
    3. Weitere Untersuchung der Anomalie erforderlich

    ✅ **Analyse abgeschlossen** - Verarbeitungszeit: 10 Sekunden"""
