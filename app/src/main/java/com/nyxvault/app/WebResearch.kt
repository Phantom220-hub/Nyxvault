package com.nyxvault.app

import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

data class WebHit(val title: String, val url: String, val snippet: String)

object WebResearch {
    fun search(query: String): List<WebHit> {
        val encoded = URLEncoder.encode(query, "UTF-8")
        val connection = (URL("https://html.duckduckgo.com/html/?q=$encoded").openConnection() as HttpURLConnection)
        connection.requestMethod = "GET"
        connection.connectTimeout = 9000
        connection.readTimeout = 12000
        connection.setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/120 Mobile Safari/537.36")
        connection.setRequestProperty("Accept-Language", "ru,en;q=0.8,tg;q=0.7")
        try {
            val html = connection.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
            val links = Regex("""<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>""", RegexOption.IGNORE_CASE)
            val snippets = Regex("""<(?:a|div)[^>]*class="result__snippet"[^>]*>(.*?)</(?:a|div)>""", RegexOption.IGNORE_CASE)
            val linkMatches = links.findAll(html).toList()
            val snippetMatches = snippets.findAll(html).toList()
            return linkMatches.take(6).mapIndexedNotNull { index, match ->
                val rawUrl = decode(match.groupValues[1])
                val url = if (rawUrl.contains("uddg=")) {
                    val target = Regex("[?&]uddg=([^&]+)").find(rawUrl)?.groupValues?.get(1)
                    target?.let { java.net.URLDecoder.decode(it, "UTF-8") } ?: rawUrl
                } else rawUrl
                if (!url.startsWith("http")) return@mapIndexedNotNull null
                WebHit(
                    decode(match.groupValues[2].replace(Regex("<.*?>"), "")).trim(),
                    url,
                    snippetMatches.getOrNull(index)?.groupValues?.get(1)?.let { decode(it.replace(Regex("<.*?>"), "")) }?.trim().orEmpty()
                )
            }.filter { it.title.isNotBlank() }
        } finally {
            connection.disconnect()
        }
    }

    private fun decode(value: String): String = value
        .replace("&amp;", "&").replace("&quot;", "\"").replace("&#x27;", "'")
        .replace("&#39;", "'").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&nbsp;", " ").replace(Regex("&#(\\d+);")) {
            it.groupValues[1].toIntOrNull()?.toChar()?.toString() ?: ""
        }
}
