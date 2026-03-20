/**
 * O.R.I.O.N. — Configuration globale
 * ------------------------------------
 * Détecte automatiquement l'environnement :
 *   - FastAPI (port 8000)  → API_BASE = ""  (même origine)
 *   - Live Server (port 5500/5501) → API_BASE = "http://localhost:8000"
 *
 * Utilisation dans les pages HTML :
 *   fetch(API_BASE + '/api/overview')
 *   new WebSocket(WS_BASE + '/ws')
 */

(function () {
  var port = location.port;
  var host = location.hostname;

  // FastAPI backend
  var BACKEND_HOST = 'localhost';
  var BACKEND_PORT = '8000';

  if (port === BACKEND_PORT) {
    // On est déjà servi par FastAPI → pas de préfixe
    window.API_BASE = '';
    window.WS_BASE = (location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host;
  } else {
    // Live Server ou autre → pointer vers le backend FastAPI
    window.API_BASE = 'http://' + BACKEND_HOST + ':' + BACKEND_PORT;
    window.WS_BASE = 'ws://' + BACKEND_HOST + ':' + BACKEND_PORT;
  }

  console.log('[ORION] Environnement détecté — port=' + port);
  console.log('[ORION] API_BASE=' + window.API_BASE);
  console.log('[ORION] WS_BASE=' + window.WS_BASE);
})();
