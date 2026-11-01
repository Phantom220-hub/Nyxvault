package com.nyxvault.app

import android.app.Activity
import android.app.AlertDialog
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = bg
        window.navigationBarColor = bg
        knowledgeFile = File(filesDir, "knowledge.jsonl")
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
            text = "PERSONAL OFFLINE AI  •  VERSION 0.1"
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
        val teacher = makeButton("TEACHER MODE").apply { textSize = 11f; setOnClickListener { openTeacherGate() } }
        topRow.addView(teacher, LinearLayout.LayoutParams(0, dp(40), 1f).apply { marginStart = dp(8) })
        root.addView(topRow)

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
        if (question.isEmpty()) return
        input.text.clear()
        addMessage(question, true)
        addMessage(findAnswer(question), false)
    }

    private fun addMessage(value: String, fromUser: Boolean) {
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
        if (!knowledgeFile.exists()) {
            assets.open("knowledge.jsonl").bufferedReader(Charsets.UTF_8).use { knowledgeFile.writeText(it.readText(), Charsets.UTF_8) }
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
        Regex("[a-zа-яё0-9_+#.]+", RegexOption.IGNORE_CASE)
            .findAll(value.lowercase(Locale.ROOT).replace('ё', 'е')).map { it.value }.toSet()

    private fun findAnswer(question: String): String {
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

    private fun updateStatus() {
        status.text = "● OFFLINE  •  ${examples.size} знаний  •  данные хранятся на устройстве"
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
