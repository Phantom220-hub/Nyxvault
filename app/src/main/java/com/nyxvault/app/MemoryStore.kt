package com.nyxvault.app

import android.content.Context
import org.json.JSONObject
import java.io.File

class MemoryStore(context: Context) {
    private val file = File(context.filesDir, "long_term_memory.jsonl")

    @Synchronized
    fun add(value: String): Boolean {
        val clean = value.trim().replace("\n", " ")
        if (clean.isBlank()) return false
        val existing = all()
        if (existing.any { it.equals(clean, ignoreCase = true) }) return false
        file.appendText(JSONObject().put("fact", clean).toString() + "\n", Charsets.UTF_8)
        return true
    }

    @Synchronized
    fun all(): List<String> {
        if (!file.exists()) return emptyList()
        return file.readLines(Charsets.UTF_8).mapNotNull {
            try { JSONObject(it).optString("fact").trim().takeIf(String::isNotEmpty) }
            catch (_: Exception) { null }
        }
    }

    @Synchronized
    fun clear() {
        file.writeText("", Charsets.UTF_8)
    }
}
