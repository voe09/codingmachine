class Solution {
    public int longestStrChain(String[] words) {
        Arrays.sort(words, (a, b) -> a.length() - b.length());
        HashMap<String, Integer> maxLen = new HashMap<>();
        int res = 1;
        for (String word : words) {
            StringBuilder sb = new StringBuilder(word);
            int curLen = 1;
            for (int i = 0; i < word.length(); i++) {
                sb.deleteCharAt(i);
                String pre = sb.toString();
                curLen = Math.max(curLen, maxLen.getOrDefault(pre, 0) + 1);
                sb.insert(i, word.charAt(i));
            }
            maxLen.put(word, curLen);
            res = Math.max(res, curLen);
        }
        return res;
    }
}