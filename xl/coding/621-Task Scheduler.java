// O(N)
class Solution {
    public int leastInterval(char[] tasks, int n) {
        int len = tasks.length;
        char[] freq = new char[26];
        int maxFreq = 0;
        int maxTaskNum = 0;
        for (char c : tasks) {
            freq[c - 'A']++;
            if (freq[c - 'A'] > maxFreq) {
                maxFreq = freq[c - 'A'];
                maxTaskNum = 1;
            } else if (freq[c - 'A'] == maxFreq) {
                maxTaskNum++;
            }
        }
        int idleSlots = n * (maxFreq - 1);
        int taskToFill = len - maxFreq - (maxTaskNum - 1);
        int idleSlotsLeft = idleSlots - taskToFill;
        return idleSlotsLeft > 0 ? len + idleSlotsLeft : len;
    }
}