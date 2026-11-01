"""Local browser interface for NYXVAULT. Binds to localhost only."""
import hmac
import json
import os
import secrets
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nyxvault import Nyxvault, load_examples, train
from neural_brain import NeuralBrain, train_neural

DATA = Path("data/knowledge.jsonl")
INDEX = Path("data/model_index.json")
MODEL = Path("data/neural_model.npz")
SESSIONS = {}


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NYXVAULT</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#080d18;color:#e8efff}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 20% 0%,#182b4a,#080d18 55%)}
main{max-width:820px;margin:auto;padding:24px 16px}.brand{font-size:30px;font-weight:800;letter-spacing:3px;color:#9cc8ff}
.sub{color:#9aabc7;margin:4px 0 20px}.panel{background:#111b2d;border:1px solid #263b5d;border-radius:16px;padding:16px;margin:14px 0}
#messages{height:48vh;min-height:260px;overflow:auto;display:flex;flex-direction:column;gap:10px}
.msg{white-space:pre-wrap;overflow-wrap:anywhere;padding:11px 14px;border-radius:12px;max-width:92%;line-height:1.45}
.user{align-self:flex-end;background:#1d4778}.bot{align-self:flex-start;background:#1b2940}
.row{display:flex;gap:8px;margin-top:10px}input,textarea{width:100%;background:#091322;border:1px solid #334967;border-radius:10px;color:#fff;padding:12px;font:inherit}
button{background:#2b69a8;color:white;border:0;border-radius:10px;padding:11px 15px;font:inherit;font-weight:650}
button:disabled{opacity:.5}textarea{min-height:76px;resize:vertical}.muted{color:#9aabc7;font-size:13px}
details summary{cursor:pointer;font-weight:700}.status{color:#93d7b0}
</style></head><body><main>
<div class="brand">NYXVAULT</div><div class="sub">Локальный ИИ · Chat Mode / Teacher Mode</div>
<section class="panel"><div id="messages"><div class="msg bot">Привет! Я NYXVAULT. Введи вопрос, чтобы начать разговор.</div></div>
<div class="row"><input id="question" placeholder="Напиши сообщение..." maxlength="2000"><button id="send">Отправить</button></div>
<div class="muted" id="status">Загрузка состояния...</div></section>
<section class="panel"><details><summary>Teacher Mode — обучение</summary>
<p class="muted">Сначала войди с паролем, заданным в NYXVAULT_TEACHER_PASSWORD. Пароль не сохраняется в базе знаний.</p>
<div class="row"><input id="password" type="password" placeholder="Пароль Teacher Mode"><button id="login">Войти</button></div>
<div id="teacher" hidden>
<p class="status">Teacher Mode разблокирован для этой сессии.</p>
<input id="teachq" placeholder="Вопрос" maxlength="1000">
<textarea id="teacha" placeholder="Проверенный ответ" maxlength="4000"></textarea>
<div class="row"><button id="save">Сохранить пример</button><button id="train">Переобучить ИИ</button></div>
</div><div id="teacherstatus" class="muted"></div>
</details></section>
<p class="muted">Сервер доступен только на этом устройстве (127.0.0.1). Chat Mode не сохраняет переписку в учебную базу.</p>
</main><script>
const $=id=>document.getElementById(id);
function add(text,who){const el=document.createElement('div');el.className='msg '+who;el.textContent=text;$('messages').appendChild(el);$('messages').scrollTop=$('messages').scrollHeight}
async function api(path,data={}){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const d=await r.json();if(!r.ok)throw new Error(d.error||'Ошибка запроса');return d}
async function status(){try{const r=await fetch('/api/status');const d=await r.json();$('status').textContent='Примеров: '+d.examples+' · RNN: '+(d.model?'обучена':'ещё не обучена');}catch(e){$('status').textContent='Не удалось получить состояние.'}}
async function send(){const q=$('question').value.trim();if(!q)return;$('question').value='';add(q,'user');$('send').disabled=true;try{const d=await api('/api/chat',{question:q});add(d.answer,'bot')}catch(e){add(e.message,'bot')}finally{$('send').disabled=false;$('question').focus()}}
$('send').onclick=send;$('question').addEventListener('keydown',e=>{if(e.key==='Enter')send()});
$('login').onclick=async()=>{try{await api('/api/teacher/login',{password:$('password').value});$('password').value='';$('teacher').hidden=false;$('teacherstatus').textContent='Вход выполнен.'}catch(e){$('teacherstatus').textContent=e.message}};
$('save').onclick=async()=>{try{const d=await api('/api/teacher/teach',{question:$('teachq').value,answer:$('teacha').value});$('teacherstatus').textContent=d.message;$('teachq').value='';$('teacha').value='';await status()}catch(e){$('teacherstatus').textContent=e.message}};
$('train').onclick=async()=>{if(!confirm('Запустить локальное переобучение? Это может занять некоторое время.'))return;$('train').disabled=true;$('teacherstatus').textContent='Обучение выполняется...';try{const d=await api('/api/teacher/train',{});$('teacherstatus').textContent='Готово: '+d.examples+' примеров; RNN loss '+d.loss;await status()}catch(e){$('teacherstatus').textContent=e.message}finally{$('train').disabled=false}};
status();
</script></body></html>"""


def session_for(handler):
    cookie = SimpleCookie()
    cookie.load(handler.headers.get("Cookie", ""))
    morsel = cookie.get("nyxvault_session")
    token = morsel.value if morsel else ""
    if token not in SESSIONS:
        token = secrets.token_urlsafe(32)
        SESSIONS[token] = {"teacher": False, "history": []}
        handler.new_session = token
    return token, SESSIONS[token]


class Handler(BaseHTTPRequestHandler):
    server_version = "NYXVAULTLocal/1.0"

    def _headers(self, status=200, content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        if getattr(self, "new_session", None):
            self.send_header("Set-Cookie", f"nyxvault_session={self.new_session}; HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()

    def _json(self, payload, status=200):
        self._headers(status)
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _body(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Некорректный размер запроса.")
        if size < 0 or size > 12000:
            raise ValueError("Запрос слишком большой.")
        raw = self.rfile.read(size)
        data = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(data, dict):
            raise ValueError("Ожидается JSON-объект.")
        return data

    def do_GET(self):
        token, session = session_for(self)
        if self.path == "/":
            self._headers(200, "text/html; charset=utf-8")
            self.wfile.write(PAGE.encode("utf-8"))
        elif self.path == "/api/status":
            self._headers(200)
            self.wfile.write(json.dumps({
                "examples": len(load_examples(DATA)),
                "model": MODEL.exists(),
                "teacher": session["teacher"],
            }).encode("utf-8"))
        else:
            self._json({"error": "Страница не найдена."}, 404)

    def do_POST(self):
        token, session = session_for(self)
        try:
            data = self._body()
            if self.path == "/api/chat":
                question = str(data.get("question", "")).strip()
                if not question or len(question) > 2000:
                    return self._json({"error": "Введи вопрос длиной до 2000 символов."}, 400)
                answer = Nyxvault(DATA, INDEX).answer(question)
                if answer.startswith("Я пока не знаю") and MODEL.exists():
                    neural = NeuralBrain(MODEL)
                    recent = session["history"][-3:]
                    context = "".join(f"Вопрос: {q}\nОтвет: {a}\n" for q, a in recent)
                    generated = neural.generate(
                        context + f"Вопрос: {question}\nОтвет:",
                        max_new=150, temperature=0.65, top_k=6
                    )
                    if generated:
                        answer = "[Экспериментальная RNN] " + generated
                session["history"].append((question, answer))
                session["history"] = session["history"][-4:]
                return self._json({"answer": answer})
            if self.path == "/api/teacher/login":
                expected = os.environ.get("NYXVAULT_TEACHER_PASSWORD")
                supplied = str(data.get("password", ""))
                if not expected:
                    return self._json({"error": "Пароль не настроен в окружении сервера."}, 503)
                if not hmac.compare_digest(supplied, expected):
                    return self._json({"error": "Неверный пароль."}, 403)
                session["teacher"] = True
                return self._json({"ok": True})
            if self.path == "/api/teacher/teach":
                if not session["teacher"]:
                    return self._json({"error": "Сначала войди в Teacher Mode."}, 403)
                question = str(data.get("question", "")).strip()
                answer = str(data.get("answer", "")).strip()
                if not question or not answer or len(question) > 1000 or len(answer) > 4000:
                    return self._json({"error": "Заполни вопрос и ответ (до 1000 и 4000 символов)."}, 400)
                Nyxvault(DATA, INDEX).teach(question, answer)
                return self._json({"ok": True, "message": "Пример сохранён. Нажми «Переобучить ИИ»."})
            if self.path == "/api/teacher/train":
                if not session["teacher"]:
                    return self._json({"error": "Сначала войди в Teacher Mode."}, 403)
                retrieval = train(DATA, INDEX)
                neural = train_neural(DATA, MODEL)
                return self._json({
                    "ok": True,
                    "examples": retrieval["examples"],
                    "loss": neural["loss"],
                })
            return self._json({"error": "Маршрут не найден."}, 404)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            return self._json({"error": str(error)}, 400)
        except OSError as error:
            return self._json({"error": f"Ошибка локального файла: {error}"}, 500)

    def log_message(self, fmt, *args):
        # Do not print submitted chat text or passwords to the terminal.
        print("NYXVAULT web:", fmt % args)


def main():
    port = int(os.environ.get("NYXVAULT_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"NYXVAULT запущен: http://127.0.0.1:{port}")
    print("Остановить сервер: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка NYXVAULT...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
