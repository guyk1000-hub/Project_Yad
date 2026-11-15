package yad.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ModelNamesTest {
    @Test
    fun `names keep the user, the label and the time`() {
        val n = ModelNames.phoneTrained("tomer", " after lunch ", "20261115_184045")
        assertEquals("tomer-after_lunch@20261115_184045", n)
        assertEquals("tomer", ModelNames.userOf(n))
        assertEquals("tomer · after_lunch · 15/11 18:40", ModelNames.display(n))
        assertTrue(ModelNames.isPhoneTrained(n))
    }

    @Test
    fun `older names still work`() {
        assertEquals("tomer", ModelNames.userOf("tomer-phone@20261115_184045"))
        assertEquals("tomer · phone · 15/11 18:40", ModelNames.display("tomer-phone@20261115_184045"))
        assertEquals("tomer", ModelNames.userOf("tomer@20261115_1758"))
        assertEquals("tomer · 15/11 17:58", ModelNames.display("tomer@20261115_1758"))
        assertFalse(ModelNames.isPhoneTrained("tomer@20261115_1758"))
        assertEquals("tomer_long", ModelNames.userOf("tomer_long"))
        assertEquals("tomer_long", ModelNames.display("tomer_long"))
    }

    @Test
    fun `labels can't break the name`() {
        assertEquals("ab", ModelNames.cleanLabel("a-@/b"))
        assertEquals("phone", ModelNames.cleanLabel("  @@ "))
        assertEquals("טומר_1", ModelNames.cleanLabel("טומר 1"))
        assertEquals(24, ModelNames.cleanLabel("x".repeat(40)).length)
    }
}
