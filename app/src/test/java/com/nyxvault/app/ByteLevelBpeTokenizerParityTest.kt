package com.nyxvault.app

import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ByteLevelBpeTokenizerParityTest {
    @Test
    fun androidTokenizerMatchesCanonicalPythonGoldenVectors() {
        val assetDir = File("src/main/assets/nyxvault_model")
        val tokenizerFile = File(assetDir, "tokenizer_mobile.json")
        val fixtureFile = File(assetDir, "tokenizer_parity_test.json")

        assertTrue("Android tokenizer asset is missing: $tokenizerFile", tokenizerFile.isFile)
        assertTrue("Golden vectors are missing: $fixtureFile", fixtureFile.isFile)

        val tokenizer = ByteLevelBpeTokenizer(tokenizerFile.readText(Charsets.UTF_8))
        val fixture = JSONObject(fixtureFile.readText(Charsets.UTF_8))
        val cases = fixture.getJSONArray("cases")
        assertTrue("Expected at least 8 multilingual tokenizer cases", cases.length() >= 8)

        for (index in 0 until cases.length()) {
            val item = cases.getJSONObject(index)
            val text = item.getString("text")
            val expectedIdsJson = item.getJSONArray("ids")
            val expectedIds = (0 until expectedIdsJson.length()).map { expectedIdsJson.getInt(it) }
            val actualIds = tokenizer.encode(text)

            assertEquals("Token IDs differ for case: $text", expectedIds, actualIds)
            assertEquals(
                "Decoded text differs for case: $text",
                item.getString("decoded"),
                tokenizer.decode(actualIds)
            )
        }
    }
}
