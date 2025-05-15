class Solution {
    public String longestDiverseString(int a, int b, int c) {
        TreeSet<Pair<Integer, Character>> tset = new TreeSet<Pair<Integer, Character>>((x, y) -> {
            if (y.getKey() - x.getKey() == 0) {
                return y.getValue() - x.getValue();
            } else {
                return y.getKey() - x.getKey();
            }
        });
        tset.add(new Pair<Integer, Character>(a, 'a'));
        tset.add(new Pair<Integer, Character>(b, 'b'));
        tset.add(new Pair<Integer, Character>(c, 'c'));
        List<Character> order = new ArrayList<Character>();
        for (Pair<Integer, Character> p : tset) {
            if (p.getKey() == 0) {
                continue;
            }
            order.add(p.getValue());
        }
        int[] count = new int[] { a, b, c };
        StringBuilder sb = new StringBuilder();
        int idx = 0;
        while (order.size() > 1) {
            char cur = order.get(idx);
            if (count[cur - 'a'] == 0) {
                idx = (idx + 1) % order.size();
                continue;
            }
            sb.append(cur);
            if (--count[cur - 'a'] == 0) {
                order.remove(idx);
                idx = (idx + 1) % order.size();
                continue;
            }
            sb.append(cur);
            if (--count[cur - 'a'] == 0) {
                order.remove(idx);
            }
            idx = (idx + 1) % order.size();
        }
        idx = 0;
        char last = order.get(0);
        while (idx <= sb.length() && count[last - 'a'] > 0) {
            if ( (idx > 0 && sb.charAt(idx - 1) == last) 
                || (idx < sb.length() && sb.charAt(idx) == last) ) {
                idx++;
                continue;
            }
            sb.insert(idx, last);
            if (--count[last - 'a'] == 0) {
                break;
            }
            sb.insert(idx, last);
            count[last - 'a']--;
            idx += 3;
        }
        return sb.toString();
    }
}