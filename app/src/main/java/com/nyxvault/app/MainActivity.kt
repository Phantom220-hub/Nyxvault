package com.nyxvault.app

import android.app.Activity
import android.app.AlertDialog
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.content.Intent
import android.content.pm.PackageManager
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONObject
import java.io.File
import java.util.Locale
import kotlin.math.ln
import kotlin.math.sqrt

private data class Example(val question: String, val answer: String)

class MainActivity : Activity() {
    private val bg = Color.rgb(7, 12, 24)
    private val panel = Color.rgb(17, 27, 46)
    private val accent = Color.rgb(65, 143, 224)
    private val textColor = Color.rgb(232, 240, 255)
    private val muted = Color.rgb(150, 169, 198)
    private lateinit var messages: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var input: EditText
    private lateinit var status: TextView
    private lateinit var knowledgeFile: File
    private var examples = mutableListOf<Example>()
    private var teacherUnlocked = false
    private var webEnabled = true
    private var busy = false
    private lateinit var webButton: Button
    private lateinit var memoryStore: MemoryStore
    private lateinit var voiceButton: Button
    private lateinit var modeButton: Button
    private var currentMode = "Обычный"
    private var voiceOutput = false
    private var pendingSpeechLanguage = "ru-RU"
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private val speechRequestCode = 7042
    private val audioPermissionCode = 7043

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = bg
        window.navigationBarColor = bg
        knowledgeFile = File(filesDir, "knowledge.jsonl")
        memoryStore = MemoryStore(this)
        tts = TextToSpeech(this) { result ->
            ttsReady = result == TextToSpeech.SUCCESS
            if (ttsReady) tts?.language = Locale("ru", "RU")
        }
        loadKnowledge()
        buildInterface()
        addMessage("Привет! Я NYXVAULT — твой локальный ИИ. Мои ответы основаны на собственной базе знаний.\n\nСпроси, например: «Что такое Python?»", false)
        updateStatus()
    }

    private fun buildInterface() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(20), dp(18), dp(12))
            setBackgroundColor(bg)
        }
        root.addView(TextView(this).apply {
            text = "NYXVAULT"
            textSize = 29f
            letterSpacing = 0.16f
            setTextColor(Color.rgb(153, 200, 255))
            typeface = Typeface.create("sans-serif", Typeface.BOLD)
        })
        root.addView(TextView(this).apply {
            text = "PERSONAL AI  •  LOCAL BRAIN + LIVE WEB RESEARCH"
            textSize = 10f
            letterSpacing = 0.12f
            setTextColor(muted)
            setPadding(0, dp(3), 0, dp(14))
        })
        val topRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        topRow.addView(TextView(this).apply {
            text = " CHAT MODE "
            textSize = 12f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(textColor)
            background = rounded(accent, 18)
            setPadding(dp(12), dp(10), dp(12), dp(10))
        })
        val teacher = makeButton("TEACHER").apply { textSize = 10f; setOnClickListener { openTeacherGate() } }
        topRow.addView(teacher, LinearLayout.LayoutParams(0, dp(40), 1f).apply { marginStart = dp(6) })
        webButton = makeButton("WEB: ON").apply { textSize = 10f; setOnClickListener {
            webEnabled = !webEnabled
            text = if (webEnabled) "WEB: ON" else "WEB: OFF"
            updateStatus()
        } }
        topRow.addView(webButton, LinearLayout.LayoutParams(0, dp(40), 1f).apply { marginStart = dp(6) })
        root.addView(topRow)

        val toolsRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        modeButton = makeButton("MODE: ОБЫЧНЫЙ").apply { textSize = 9f; setOnClickListener { chooseMode() } }
        toolsRow.addView(modeButton, LinearLayout.LayoutParams(0, dp(38), 1f).apply { topMargin = dp(7); marginEnd = dp(5) })
        val memory = makeButton("MEMORY").apply { textSize = 10f; setOnClickListener { showMemory() } }
        toolsRow.addView(memory, LinearLayout.LayoutParams(0, dp(38), 1f).apply { topMargin = dp(7); marginEnd = dp(5) })
        voiceButton = makeButton("VOICE").apply { textSize = 10f; setOnClickListener {
            AlertDialog.Builder(this@MainActivity).setTitle("Голосовой режим")
                .setItems(arrayOf("🎙 Распознать речь", if (voiceOutput) "Отключить озвучивание" else "Включить озвучивание")) { _, which ->
                    if (which == 0) chooseSpeechLanguage() else {
                        voiceOutput = !voiceOutput
                        voiceButton.text = if (voiceOutput) "VOICE: ON" else "VOICE"
                        if (!voiceOutput) tts?.stop()
                    }
                }.show()
        } }
        toolsRow.addView(voiceButton, LinearLayout.LayoutParams(0, dp(38), 1f).apply { topMargin = dp(7) })
        root.addView(toolsRow)

        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = rounded(panel, 20)
            setPadding(dp(12), dp(12), dp(12), dp(10))
        }
        scroll = ScrollView(this)
        messages = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(2), dp(2), dp(2), dp(8))
        }
        scroll.addView(messages)
        card.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(card, LinearLayout.LayoutParams(-1, 0, 1f).apply { topMargin = dp(14) })

        status = TextView(this).apply {
            textSize = 11f
            setTextColor(muted)
            setPadding(dp(3), dp(10), dp(3), dp(8))
        }
        root.addView(status)

        val composer = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        input = EditText(this).apply {
            hint = "Напиши сообщение..."
            setHintTextColor(muted)
            setTextColor(textColor)
            textSize = 15f
            maxLines = 1
            imeOptions = EditorInfo.IME_ACTION_SEND
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            background = rounded(Color.rgb(24, 38, 61), 14)
            setPadding(dp(14), 0, dp(14), 0)
            setOnEditorActionListener { _, action, _ ->
                if (action == EditorInfo.IME_ACTION_SEND) { sendMessage(); true } else false
            }
        }
        composer.addView(input, LinearLayout.LayoutParams(0, dp(52), 1f))
        val send = makeButton("➤").apply { textSize = 20f; setOnClickListener { sendMessage() } }
        composer.addView(send, LinearLayout.LayoutParams(dp(54), dp(52)).apply { marginStart = dp(8) })
        root.addView(composer)
        setContentView(root)
    }

    private fun sendMessage() {
        val question = input.text.toString().trim()
        if (question.isEmpty() || busy) return
        input.text.clear()
        addMessage(question, true)
        if (handleMemoryCommand(question)) return
        var local = findAnswer(question)
        val questionTokens = tokenize(question)
        val relevantMemory = memoryStore.all().filter { fact ->
            tokenize(fact).intersect(questionTokens).size >= 2
        }.take(3)
        if (relevantMemory.isNotEmpty()) {
            local += "\n\nКОНТЕКСТ ИЗ ТВОЕЙ ПАМЯТИ:\n" + relevantMemory.joinToString("\n")
        }
        if (!webEnabled) {
            addMessage(local, false)
            return
        }
        busy = true
        status.text = "● NYXVAULT THINKING  •  ищу свежие источники..."
        Thread {
            val searchQuery = when (currentMode) {
                "Учитель" -> "$question пошаговое объяснение учебный материал"
                "Разработчик" -> "$question programming documentation code examples"
                "Исследователь" -> "$question research scientific sources"
                "Детектив" -> "$question logical analysis evidence contradictions"
                else -> question
            }
            val hits = try { WebResearch.search(searchQuery) } catch (_: Exception) { emptyList() }
            runOnUiThread {
                val response = if (hits.isEmpty()) {
                    "$local\n\nНе удалось получить свежие данные из интернета. Проверь подключение или выключи WEB."
                } else {
                    val sources = hits.take(5).joinToString("\n\n") { "• ${it.title}\n${it.snippet}\n${it.url}" }
                    "РЕЗУЛЬТАТ ИССЛЕДОВАНИЯ\nЯ нашёл материалы в интернете. Сверяй важные факты по первоисточникам.\n\nЛокальная база: $local\n\nИСТОЧНИКИ\n$sources\n\nВажно: это найденные страницы и краткие фрагменты, а не гарантия истинности. Для спорных тем нужно сравнивать несколько независимых источников."
                }
                addMessage(response, false)
                busy = false
                updateStatus()
            }
        }.start()
    }

    private fun addMessage(value: String, fromUser: Boolean) {
        if (!fromUser && voiceOutput && ttsReady) {
            tts?.speak(value.take(2500), TextToSpeech.QUEUE_FLUSH, null, "nyxvault_reply")
        }
        val bubble = TextView(this).apply {
            text = value
            textSize = 14f
            setTextColor(textColor)
            setPadding(dp(13), dp(10), dp(13), dp(10))
            background = rounded(if (fromUser) Color.rgb(32, 76, 124) else Color.rgb(28, 43, 67), 16)
        }
        val row = LinearLayout(this).apply {
            gravity = if (fromUser) Gravity.END else Gravity.START
            setPadding(0, dp(5), 0, dp(5))
            addView(bubble, LinearLayout.LayoutParams((resources.displayMetrics.widthPixels * 0.78f).toInt(), -2))
        }
        messages.addView(row)
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun loadKnowledge() {
        val bundled = assets.open("knowledge.jsonl").bufferedReader(Charsets.UTF_8).use { it.readLines() }
        if (!knowledgeFile.exists()) {
            knowledgeFile.writeText(bundled.joinToString("\n", postfix = "\n"), Charsets.UTF_8)
        } else {
            // Upgrade existing installs without overwriting anything the user learned locally.
            val knownQuestions = knowledgeFile.readLines(Charsets.UTF_8).mapNotNull { line ->
                try { JSONObject(line).optString("question").trim().lowercase(Locale.ROOT) } catch (_: Exception) { null }
            }.toMutableSet()
            val additions = bundled.filter { line ->
                try {
                    val q = JSONObject(line).optString("question").trim().lowercase(Locale.ROOT)
                    q.isNotEmpty() && knownQuestions.add(q)
                } catch (_: Exception) { false }
            }
            if (additions.isNotEmpty()) knowledgeFile.appendText(additions.joinToString("\n", prefix = "\n", postfix = "\n"), Charsets.UTF_8)
        }
        examples = knowledgeFile.readLines(Charsets.UTF_8).mapNotNull { line ->
            try {
                val row = JSONObject(line)
                val q = row.optString("question").trim()
                val a = row.optString("answer").trim()
                if (q.isNotEmpty() && a.isNotEmpty()) Example(q, a) else null
            } catch (_: Exception) { null }
        }.toMutableList()
    }

    private fun tokenize(value: String): Set<String> =
        Regex("[a-zа-яёӯғқҳҷӣ0-9_+#.]+", RegexOption.IGNORE_CASE)
            .findAll(value.lowercase(Locale.ROOT).replace('ё', 'е')).map { it.value }.toSet()

    private fun findAnswer(question: String): String {
        MathSolver.solve(question)?.let { return it }
        val q = tokenize(question)
        if (q.isEmpty()) return "Напиши вопрос — я попробую найти подходящий ответ."
        val docs = examples.map { tokenize(it.question) }
        val n = maxOf(1, docs.size)
        val vocab = docs.flatten().toSet()
        val queryWords = q.intersect(vocab)
        if (queryWords.isEmpty()) return "Я пока не знаю этого. Открой Teacher Mode и обучи меня правильному ответу."
        val queryWeights = queryWords.associateWith { word -> ln(1.0 + n.toDouble() / (1.0 + docs.count { word in it })) }
        val queryNorm = sqrt(queryWeights.values.sumOf { it * it }).takeIf { it > 0.0 } ?: 1.0
        var bestIndex = -1
        var bestScore = 0.0
        for (i in examples.indices) {
            val doc = docs[i]
            val weights = doc.associateWith { word -> ln(1.0 + n.toDouble() / (1.0 + docs.count { word in it })) }
            val docNorm = sqrt(weights.values.sumOf { it * it }).takeIf { it > 0.0 } ?: 1.0
            val score = queryWeights.entries.sumOf { (word, weight) ->
                (weight / queryNorm) * ((weights[word] ?: 0.0) / docNorm)
            }
            if (score > bestScore) { bestScore = score; bestIndex = i }
        }
        return if (bestIndex >= 0 && bestScore >= 0.16) examples[bestIndex].answer
        else "Я пока не знаю точного ответа. Добавь проверенный пример через Teacher Mode."
    }

    private fun openTeacherGate() {
        val prefs = getSharedPreferences("nyxvault", MODE_PRIVATE)
        val saved = prefs.getString("teacher_password", null)
        val field = EditText(this).apply {
            hint = if (saved == null) "Создай пароль (минимум 6 символов)" else "Пароль Teacher Mode"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        AlertDialog.Builder(this)
            .setTitle(if (saved == null) "Настройка Teacher Mode" else "Вход в Teacher Mode")
            .setMessage(if (saved == null) "Пароль хранится только на этом телефоне." else "Введи свой пароль.")
            .setView(field)
            .setNegativeButton("Отмена", null)
            .setPositiveButton(if (saved == null) "Создать" else "Войти") { _, _ ->
                val value = field.text.toString()
                if (saved == null) {
                    if (value.length < 6) toast("Пароль должен содержать минимум 6 символов.")
                    else {
                        prefs.edit().putString("teacher_password", value).apply()
                        teacherUnlocked = true
                        showTeacherEditor()
                    }
                } else if (value == saved) {
                    teacherUnlocked = true
                    showTeacherEditor()
                } else toast("Неверный пароль.")
            }.show()
    }

    private fun showTeacherEditor() {
        if (!teacherUnlocked) return
        val box = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(20), dp(8), dp(20), 0) }
        val q = EditText(this).apply { hint = "Вопрос"; setTextColor(textColor) }
        val a = EditText(this).apply { hint = "Проверенный ответ"; minLines = 4; gravity = Gravity.TOP; setTextColor(textColor) }
        box.addView(q)
        box.addView(a)
        AlertDialog.Builder(this)
            .setTitle("Обучить NYXVAULT")
            .setMessage("Добавь вопрос и правильный ответ. Он сохранится на этом устройстве.")
            .setView(box)
            .setNegativeButton("Закрыть", null)
            .setNeutralButton("Выйти") { _, _ -> teacherUnlocked = false; toast("Teacher Mode заблокирован.") }
            .setPositiveButton("Сохранить") { _, _ ->
                val question = q.text.toString().trim()
                val answer = a.text.toString().trim()
                if (question.isBlank() || answer.isBlank()) toast("Заполни оба поля.")
                else try {
                    knowledgeFile.appendText(JSONObject().put("question", question).put("answer", answer).toString() + "\n", Charsets.UTF_8)
                    loadKnowledge()
                    updateStatus()
                    toast("Знание сохранено и добавлено в индекс.")
                } catch (e: Exception) { toast("Не удалось сохранить: ${e.message}") }
            }.show()
    }

    private fun handleMemoryCommand(question: String): Boolean {
        val q = question.trim()
        val lower = q.lowercase(Locale.ROOT)
        val prefixes = listOf("запомни:", "запомни ", "remember:")
        val prefix = prefixes.firstOrNull { lower.startsWith(it) }
        if (prefix != null) {
            val fact = q.substring(prefix.length).trim()
            if (fact.isBlank()) addMessage("Напиши факт после команды. Например: Запомни: я изучаю C++.", false)
            else addMessage(if (memoryStore.add(fact)) "Сохранил в локальную память: $fact" else "Этот факт уже есть в памяти.", false)
            return true
        }
        if (listOf("что ты помнишь", "покажи память", "мои заметки", "что сохранено").any { lower.contains(it) }) {
            val facts = memoryStore.all()
            addMessage(if (facts.isEmpty()) "Пока память пуста. Напиши: «Запомни: ...»" else "МОЯ ДОЛГОВРЕМЕННАЯ ПАМЯТЬ\\n" + facts.mapIndexed { i, v -> "${i + 1}. $v" }.joinToString("\n"), false)
            return true
        }
        return false
    }

    private fun showMemory() {
        val facts = memoryStore.all()
        val content = EditText(this).apply {
            setText(facts.mapIndexed { i, v -> "${i + 1}. $v" }.joinToString("\\n"))
            setTextColor(textColor)
            minLines = 5
            gravity = Gravity.TOP
            hint = "Локальная память NYXVAULT"
            isFocusable = false
        }
        AlertDialog.Builder(this)
            .setTitle("LONG-TERM MEMORY")
            .setMessage("Факты хранятся только на этом устройстве. Добавляй через чат: Запомни: ...")
            .setView(content)
            .setNeutralButton("Очистить память") { _, _ ->
                AlertDialog.Builder(this).setMessage("Удалить все сохранённые факты?").setNegativeButton("Отмена", null)
                    .setPositiveButton("Удалить") { _, _ -> memoryStore.clear(); toast("Локальная память очищена.") }.show()
            }
            .setPositiveButton("Закрыть", null).show()
    }

    private fun chooseMode() {
        val modes = arrayOf("Обычный", "Учитель", "Разработчик", "Исследователь", "Детектив")
        AlertDialog.Builder(this).setTitle("Режим NYXVAULT").setItems(modes) { _, which ->
            currentMode = modes[which]
            modeButton.text = "MODE: " + currentMode.uppercase(Locale.ROOT)
            updateStatus()
        }.show()
    }

    private fun chooseSpeechLanguage() {
        AlertDialog.Builder(this).setTitle("Язык распознавания")
            .setItems(arrayOf("Русский", "Тоҷикӣ")) { _, which ->
                val locale = if (which == 1) Locale("tg", "TJ") else Locale("ru", "RU")
                val result = tts?.setLanguage(locale)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    toast("Синтез речи для выбранного языка не поддерживается этим TTS. Распознавание всё равно можно попробовать.")
                }
                startVoiceInput(if (which == 1) "tg-TJ" else "ru-RU")
            }.show()
    }

    private fun startVoiceInput(language: String) {
        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            pendingSpeechLanguage = language
            requestPermissions(arrayOf(android.Manifest.permission.RECORD_AUDIO), audioPermissionCode)
            return
        }
        if (!packageManager.queryIntentActivities(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH), 0).isNullOrEmpty()) {
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, language)
                putExtra(RecognizerIntent.EXTRA_PROMPT, if (language.startsWith("tg")) "Сӯҳбат кунед" else "Говори")
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            }
            startActivityForResult(intent, speechRequestCode)
        } else toast("На телефоне нет доступного сервиса распознавания речи.")
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == audioPermissionCode && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startVoiceInput(pendingSpeechLanguage)
        } else if (requestCode == audioPermissionCode) toast("Доступ к микрофону не предоставлен.")
    }

    @Deprecated("Deprecated in Android API, kept for compatibility")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == speechRequestCode && resultCode == RESULT_OK) {
            val spoken = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)?.firstOrNull()
            if (!spoken.isNullOrBlank()) { input.setText(spoken); sendMessage() }
        }
    }

    override fun onDestroy() {
        tts?.stop()
        tts?.shutdown()
        super.onDestroy()
    }

    private fun updateStatus() {
        status.text = "● ${if (webEnabled) "WEB ON" else "OFFLINE"} • ${currentMode.uppercase(Locale.ROOT)} • ${examples.size} знаний • ${if (busy) "анализ..." else "готов"}"
    }

    private fun makeButton(label: String) = Button(this).apply {
        text = label
        isAllCaps = false
        textSize = 14f
        setTextColor(Color.WHITE)
        background = rounded(accent, 14)
        minHeight = 0
        minimumHeight = 0
        setPadding(dp(10), 0, dp(10), 0)
    }

    private fun rounded(color: Int, radius: Int) = GradientDrawable().apply {
        setColor(color)
        cornerRadius = dp(radius).toFloat()
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
    private fun toast(value: String) = android.widget.Toast.makeText(this, value, android.widget.Toast.LENGTH_SHORT).show()
}
