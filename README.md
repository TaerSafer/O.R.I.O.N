# O.R.I.O.N.

**Omniscient Risk-Integrated Orchestration Network**
*Multi-Asset Trading Algorithm — Aladdin Philosophy*

> RedRock Capital — Baptiste de Romanet de Beaune

---

## Architecture

```
ORION/
├── orion.py                    # Point d'entrée unique
│
├── data/                       # Collecte de données
│   ├── __init__.py
│   ├── collector.py            # yfinance → SQLite (35 actifs, 10 ans)
│   └── orion.db                # Base SQLite (créée au 1er lancement)
│
├── signals/                    # Moteur de signaux
│   ├── __init__.py
│   ├── engine.py               # Scoring multi-indicateurs (18 indicateurs)
│   ├── indicators.py           # Bibliothèque technique (SMA, RSI, MACD, etc.)
│   ├── economic_cycle.py       # Détection cycle économique (proxies FRED)
│   ├── morning_brief.py        # Briefing quotidien 8h
│   ├── memory.py               # Mémoire des décisions
│   ├── confidence.py           # Score de confiance global
│   └── projections.py          # Projections de prix
│
├── risk/                       # Risk Manager (Aladdin)
│   ├── __init__.py
│   ├── manager.py              # Régimes, drawdown, sizing, diversification
│   ├── black_litterman.py      # Optimisation Black-Litterman
│   ├── scenario_engine.py      # Simulation CVaR (5 scénarios macro)
│   ├── rebalancer.py           # Rééquilibrage dynamique
│   └── macro_cache.py          # Cache pré-calcul macro (1h)
│
├── execution/                  # Broker
│   ├── __init__.py
│   └── broker.py               # IBKR paper trading + monitor Aladdin
│
├── backtest/                   # Backtesting
│   ├── __init__.py
│   └── engine.py               # Backtest bar-by-bar avec Aladdin
│
├── dashboard/                  # Interface web
│   ├── __init__.py
│   ├── app.py                  # FastAPI (REST + WebSocket)
│   └── templates/
│       ├── index.html          # Dashboard principal
│       ├── landing.html        # Page d'accueil animée
│       ├── public.html         # Page publique (métriques)
│       └── presentation.html   # Présentation investisseurs
│
├── journal/                    # Journal de trading
│   ├── __init__.py
│   └── logger.py               # Trades, événements, snapshots, notes
│
├── static/                     # Assets statiques (images, CSS custom)
│
├── requirements.txt            # Dépendances Python
├── .gitignore
├── .vscode/                    # Configuration VS Code
│   ├── settings.json           # Python + Live Server
│   ├── launch.json             # Configurations de debug
│   └── extensions.json         # Extensions recommandées
└── README.md
```

---

## Installation

```bash
# 1. Cloner ou copier le dossier ORION
cd ORION

# 2. Créer un environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Ouvrir dans VS Code
code .
```

---

## Utilisation

### Lancement complet
```bash
python orion.py              # Init + collecte 10 ans + dashboard
python orion.py --fast       # Mise à jour rapide (5 jours)
python orion.py --status     # Afficher l'état du système
```

### Dashboard seul (avec hot-reload)
```bash
uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --reload
```
Puis ouvrir : **http://localhost:8000**

### Pages disponibles
| URL | Description |
|-----|-------------|
| `/` | Landing page animée |
| `/dashboard` | Dashboard principal (temps réel) |
| `/public` | Page publique (métriques clés) |
| `/presentation` | Présentation investisseurs |

### Live Server (HTML statique)
Pour éditer les templates HTML en temps réel :
1. Ouvrir le dossier `ORION` dans VS Code
2. Installer l'extension **Live Server**
3. Clic droit sur un fichier `.html` dans `dashboard/templates/`
4. **"Open with Live Server"**

> **Note :** Live Server sert les fichiers HTML statiques. Les appels API
> (`/api/overview`, etc.) nécessitent le backend FastAPI actif en parallèle.

### Backtest
```bash
python backtest/engine.py
```

### Modules individuels
```bash
python data/collector.py       # Collecte des données
python signals/engine.py       # Scan des signaux
python risk/manager.py         # Risk manager
python risk/black_litterman.py # Allocation Black-Litterman
```

---

## VS Code — Debug

Ouvrir **Run & Debug** (Ctrl+Shift+D) et choisir parmi :
- 🚀 **ORION — Lancement complet**
- 📊 **Dashboard seul (FastAPI)** — avec hot-reload
- 📈 **Backtest**
- 📡 **Signal Engine**
- 🛡️ **Risk Manager**
- 📥 **Data Collector**
- ℹ️ **Status système**

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Backend | Python 3.11+, FastAPI, uvicorn |
| Data | yfinance, SQLite (WAL mode) |
| Risk | NumPy, SciPy (Black-Litterman, CVaR) |
| Broker | ib_insync (Interactive Brokers) |
| Frontend | HTML/CSS/JS vanilla, WebSocket |
| Fonts | Cormorant Garamond, Inter |

---

*O.R.I.O.N. v5.0 — RedRock Capital*
