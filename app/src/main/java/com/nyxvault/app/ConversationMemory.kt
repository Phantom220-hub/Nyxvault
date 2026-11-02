package com.nyxvault.app

import android.content.Context
import org.json.JSONObject
import java.io.File

/** Persistent rolling chat history. Keeps the latest 40 messages locally. */
class ConversationMemory(context: Context) {
    private val file = File(context.filesDir, "conversation_history.jsonl")

    @Synchronized
    fun add(text: String, fromUser: Boolean) {
        val clean = text.trim().take(3000)
        if (clean.isBlank()) return
        val rows = read().toMutableList()
        rows.add(JSONObject().put("text", clean).put("user", fromUser).put("time", System.currentTimeMillis()))
        val kept = rows.takeLast(40)
        file.writeText(kept.joinToString("\n", postfix = "\n") { it.toString() }, Charsets.UTF_8)
    }

    @Synchronized
    fun recent(): List<Pair<Boolean, String>> = read().mapNotNull {
        try { Pair(it.optBoolean("user"), it.optString("text")) } catch (_: Exception) { null }
    }

    @Synchronized
    fun lastUserMessage(): String = recent().lastOrNull { it.first }?.second.orEmpty()

    @Synchronized
    fun clear() { file.writeText("", Charsets.UTF_8) }

    private fun read(): List<JSONObject> {
        if (!file.exists()) return emptyList()
        return file.readLines(Charsets.UTF_8).mapNotNull {
            try { JSONObject(it) } catch (_: Exception) { null }
        }
    }
}
