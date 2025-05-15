// O( (M + N) logN )
class Solution {
    private class UnionFind {
        // a, (b, x)
        // a = x这么多个b
        HashMap<String, Pair<String, Double>> parent;

        public UnionFind(List<List<String>> equations) {
            parent = new HashMap<String, Pair<String, Double>>();
            for (List<String> list : equations) {
                String a = list.get(0);
                String b = list.get(1);
                parent.put(a, new Pair<String, Double>(a, 1.0));
                parent.put(b, new Pair<String, Double>(b, 1.0));
            }
        }

        public Pair<String, Double> find(String a) {
            if (!parent.containsKey(a)) {
                return null;
            }
            if (parent.get(a).getKey().equals(a)) {
                return parent.get(a);
            }
            Pair<String, Double> p = find(parent.get(a).getKey());
            // update weight
            Pair<String, Double> cur = new Pair<String, Double>(p.getKey(), p.getValue() * parent.get(a).getValue());
            parent.put(a, cur);
            return cur;
        }

        public boolean union(String a, String b, double wei) {
            // a / b = wei
            Pair<String, Double> pa = find(a);
            Pair<String, Double> pb = find(b);
            if (pa.getKey().equals(pb.getKey())) {
                return false;
            }
            // a = x * pa
            // b = y * pb
            // a = wei * b
            // -> pa = wei * y / x * pb
            parent.put(pa.getKey(), new Pair<String, Double>(pb.getKey(), pb.getValue() * wei / pa.getValue()));
            return true;
        }
    }

    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        UnionFind uf = new UnionFind(equations);
        for (int i = 0; i < equations.size(); i++) {
            String a = equations.get(i).get(0);
            String b = equations.get(i).get(1);
            double wei = values[i];
            uf.union(a, b, wei);
        }
        double[] res = new double[queries.size()];
        for (int i = 0; i < res.length; i++) {
            String a = queries.get(i).get(0);
            String b = queries.get(i).get(1);
            Pair<String, Double> pa = uf.find(a);
            Pair<String, Double> pb = uf.find(b);
            if ( pa == null || pb == null || !pa.getKey().equals(pb.getKey()) ) {
                res[i] = -1;
                continue;
            }
            res[i] = pa.getValue() / pb.getValue();
        }
        return res;
    }
}