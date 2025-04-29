// O(N * M): N - words count, M - average len of words
// build graph时只检查相邻word
class Solution {
    public String alienOrder(String[] words) {
        StringBuilder res = new StringBuilder();
        int[] inDegree = new int[26];
        HashMap<Character, Set<Character>> neis = new HashMap<Character, Set<Character>>();
        // build graph
        for (String word : words) {
            for (char c : word.toCharArray()) {
                neis.computeIfAbsent(c, k -> new HashSet<Character>());
            }
        }
        for (int i = 1; i < words.length; i++) {
            if (words[i - 1].length() > words[i].length() && words[i - 1].startsWith(words[i])) {
                return "";
            }
            int len = Math.min(words[i - 1].length(), words[i].length());
            for (int j = 0; j < len; j++) {
                char a = words[i - 1].charAt(j);
                char b = words[i].charAt(j);

                if (a == b) {
                    continue;
                }
                if (!neis.get(a).contains(b)) {
                    neis.get(a).add(b);
                    inDegree[b - 'a']++;
                }
                break;
            }
        }

        
        Queue<Character> q = new ArrayDeque<Character>();
        for (int i = 0; i < 26; i++) {
            if (inDegree[i] == 0 && neis.containsKey((char)(i + 'a'))) {
                q.offer((char)(i + 'a'));
            }
        }
        while (!q.isEmpty()) {
            char c = q.poll();
            res.append(c);
            if (!neis.containsKey(c)) {
                continue;
            }
            for (char cur : neis.get(c)) {
                inDegree[cur - 'a']--;
                if (inDegree[cur - 'a'] == 0) {
                    q.offer(cur);
                }
            }
        }
        for (int degree : inDegree) {
            if (degree > 0) {
                return "";
            }
        }
        return res.toString();
    }

    private void buildGraph(List<String> strL, int[] inDegree, HashMap<Character, Set<Character>> neis) {
        HashMap<Character, List<String>> followers = new HashMap<Character, List<String>>();
        char prev = strL.get(0).charAt(0);
        for (String str : strL) {
            char c = str.charAt(0);
            neis.computeIfAbsent(c, k -> new HashSet<Character>());
            if (c != prev) {
                Set<Character> nei = neis.get(prev);
                if (!nei.contains(c)) {
                    nei.add(c);
                    inDegree[c - 'a']++;
                }
                prev = c;
            }
            if (str.length() > 1) {
                followers.computeIfAbsent(c, k -> new ArrayList<String>()).add(str.substring(1));
            } else if (str.length() == 1 && followers.containsKey(c)) {
                inDegree[26]++;
            }
        }
        for (List<String> list : followers.values()) {
            buildGraph(list, inDegree, neis);
        }
    }
}