package ca.uwaterloo.cs451.project;

import io.bespin.java.util.Tokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueLineRecordReader;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.TaskCounter;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.util.LineReader;
import org.apache.log4j.Logger;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;
import tl.lin.data.pair.*;
import tl.lin.data.array.*;
import java.util.*;
import java.io.*;

public class TrainRandomForest extends Configured implements Tool {
  private static final Logger LOG = Logger.getLogger(TrainRandomForest.class);

  private static final class MyMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
    //start parsing data
    private static Text delimiter;
    private static IntArrayWritable exclude;
    private static IntWritable yidx;
    private static Text[] labels;
    private static final ArrayListWritable<DoubleArrayWritable> data = new ArrayListWritable<>();
    @Override
    public void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      delimiter = new Text(conf.get("delimiter"));
      String e0 = conf.get("exclude");
      String[] e1;
      if (e0==null) e1 = new String[0];
      else e1=e0.split(",");
      int[] e2 = new int[e1.length+1];
      for (int i=0; i<e1.length; i++) e2[i]=Integer.parseInt(e1[i]);
      yidx = new IntWritable(conf.getInt("yidx",1));
      e2[e1.length]=conf.getInt("yidx",1);
      Arrays.sort(e2);
      exclude = new IntArrayWritable(e2);
      String[] l = conf.get("labels").split(",");
      labels = new Text[l.length];
      for (int i=0; i<l.length; i++) labels[i] = new Text(l[i]);
    }
    @Override
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

      String[] tokens = value.toString().split(delimiter.toString());
      DoubleArrayWritable xy = new DoubleArrayWritable(new double[tokens.length-exclude.size()+1]);
      int idx = 0;
      for (int i=0; i<tokens.length; i++) {
        if (Arrays.binarySearch(exclude.getArray(),i)<0) {
          xy.set(idx,Double.parseDouble(tokens[i]));
          idx++;
        }
      }
      for (int i=0; i<labels.length; i++) {
        if (tokens[yidx.get()].equals(labels[i].toString())) {
          xy.set(tokens.length-exclude.size(),(double)i);
          break;
        }
      }
      data.add(xy);
    }
    //end parsing data
    //start tree building
    private static int maxdepth;
    private static int minnodes;
    private static int n;
    private static int d;

    class Node {
      int feature; //if leaf, -1
      double split; //if leaf, decision
      public Node(int f, double s) {
        feature=f;
        split=s;
      }
      @Override
      public String toString() {
        return Integer.toString(feature)+","+Double.toString(split);
      }
    }

    private static HashMap<Long,Node> tree;

    class Pair implements Comparable<Pair> {
      double x, y;
      public Pair(double X, double Y) {
        x=X;
        y=Y;
      }
      @Override
      public int compareTo(Pair p) {
        return new Double(this.x).compareTo(p.x);
      }
    }
    public double entropy(int[] c) {
      double ent = 0.0;
      int tot = 0;
      for (int i : c) {
        tot+=i;
      }
      for (int i : c) {
        if (i==0) continue;
        ent-=(double)i/tot*Math.log((double)i/tot);
      }
      return ent;
    }
    public void buildTree(int[] xiidx, int[] xjidx, long k, int depth, double entropy0) {
      if (depth==maxdepth || entropy0==0) {
        int[] c = new int[labels.length];
        for (int i : xiidx) {
          c[(int)data.get(i).get(d)]++;
        }
        int idx = 0;
        for (int i=1; i<c.length; i++) {
          if (c[idx]<c[i]) idx=i;
        }
        tree.put(k,new Node(-1,idx));
        return;
      }
      double max = 0; //of infogain
      Node argmax = null;
      int splitidx = 0;
      double entropy1 = 0.0, entropy2 = 0.0;
      double argmax1 = 0.0, argmax2 = 0.0; //of entropy
      int l = xiidx.length;
      for (int j : xjidx) {
        Pair[] xy = new Pair[l];
        int[] ytot = new int[labels.length];
        int[] ysum = new int[labels.length];
        int[] ydiff = new int[labels.length]; //ytot-ysum
        for (int i=0; i<l; i++) {
          xy[i]=new Pair(data.get(xiidx[i]).get(j),data.get(xiidx[i]).get(d));
          ytot[(int)xy[i].y]++;
        }
        Arrays.sort(xy);
        for (int i=1; i<l-minnodes+1; i++) {
          ysum[(int)xy[i-1].y]++;
          if (i<minnodes || xy[i-1].x==xy[i].x) continue;
          for (int h=0; h<labels.length; h++) ydiff[h]=ytot[h]-ysum[h];
          entropy1=entropy(ysum);
          entropy2=entropy(ydiff);
          double infogain = entropy0 - (double)i/l * entropy1 - (double)(l-i)/l * entropy2;
          if (infogain>max) {
            max=infogain;
            argmax = new Node(j,(xy[i-1].x+xy[i].x)/2);
            argmax1=entropy1;
            argmax2=entropy2;
            splitidx = i;
          }
        }
      }
      if (argmax==null) {
        int[] c = new int[labels.length];
        for (int i : xiidx) {
          c[(int)data.get(i).get(d)]++;
        }
        int idx = 0;
        for (int i=1; i<c.length; i++) {
          if (c[idx]<c[i]) idx=i;
        }
        tree.put(k,new Node(-1,idx));
        return;
      }
      tree.put(k,argmax);
      int[] x1 = new int[splitidx];
      int[] x2 = new int[l-splitidx];
      int idx1 = 0, idx2 = 0;
      for (int i=0; i<l; i++) {
        if (data.get(xiidx[i]).get(argmax.feature)<argmax.split) {
          x1[idx1]=xiidx[i];
          idx1++;
        }
        else {
          x2[idx2]=xiidx[i];
          idx2++;
        }
      }
      buildTree(x1,xjidx,2*k,depth+1,argmax1);
      buildTree(x2,xjidx,2*k+1,depth+1,argmax2);
      return;
    }

    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();

      n = data.size();
      int sample = conf.getInt("sample",0);
      if (sample==0) sample=data.size();

      d = data.get(0).size()-1;
      int features = conf.getInt("features",0);
      if (features==0) features=d;

      maxdepth = conf.getInt("maxdepth",0);
      minnodes = conf.getInt("minnodes",0);
      long seed = conf.getLong("seed",0);
      Random r = (seed==0) ? new Random() : new Random(seed);

      for (int h=0; h<conf.getInt("numtrees",0); h++) {

        int[] xjidx = new int[features]; //indices of features used in tree
        int idx = 0;
        for (int j=0; j<d; j++) { //SRSWOR from features
          if (r.nextDouble() < (double)(features-idx)/(d-j)) {
            xjidx[idx]=j;
            idx++;
          }
        }

        int[] xiidx = new int[sample]; //indices of observations used in tree
        int[] ysum = new int[labels.length];
        for (int i=0; i<sample; i++) { //SRSWR from observations
          xiidx[i]=r.nextInt()%n;
          if (xiidx[i]<0) xiidx[i]+=n;
          ysum[(int)data.get(xiidx[i]).get(d)]++;
        }

        tree = new HashMap<>();

        buildTree(xiidx,xjidx,1,1,entropy(ysum));
        context.write(new IntWritable(r.nextInt()), new Text(tree.toString()));
      }
    }
    //end tree building
  }

  public static final class MyReducer extends Reducer<IntWritable, Text, IntWritable, Text> {

    @Override
    public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      for (Text i : values) {
        context.write(key, i);
      }
    }
  }



  private TrainRandomForest() {}

  private static final class Args {
    // io
    @Option(name = "-input", metaVar = "[path]", required = true, usage = "input path")
    String input;

    @Option(name = "-output", metaVar = "[path]", required = true, usage = "output path")
    String output;

    @Option(name = "-reducers", metaVar = "[num]", usage = "number of reducers")
    int numReducers = 1;

    // parse
    @Option(name = "-delimiter", metaVar = "[string]", usage = "delimiter to split lines")
    String delimiter = ",";

    @Option(name = "-exclude", metaVar = "[string]", usage = "columns to exclude for features")
    String exclude = "";

    @Option(name = "-yidx", metaVar = "[num]", usage = "index of response")
    int yidx = 1;

    @Option(name = "-labels", metaVar = "[string]", usage = "name of labels")
    String labels = "0,1";

    // tree params
    @Option(name = "-seed", metaVar = "[num]", usage = "initial seed of RNG")
    long seed = 0;

    @Option(name = "-sample", metaVar = "[num]", usage = "bootstrap sample size per tree")
    int sample = 0;

    @Option(name = "-features", metaVar = "[num]", usage = "number of features per tree")
    int features = 0;

    @Option(name = "-numtrees", metaVar = "[num]", usage = "number of trees per mapper")
    int numtrees = 5;

    @Option(name = "-maxdepth", metaVar = "[num]", usage = "maximum depth per tree")
    int maxdepth = 64; //must be between 1-64

    @Option(name = "-minnodes", metaVar = "[num]", usage = "minimum nodes per leaf")
    int minnodes = 1;


  }


  @Override
  public int run(String[] argv) throws Exception {
    final Args args = new Args();
    CmdLineParser parser = new CmdLineParser(args, ParserProperties.defaults().withUsageWidth(100));

    try {
      parser.parseArgument(argv);
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      return -1;
    }

    LOG.info("Tool: " + TrainRandomForest.class.getSimpleName());
    LOG.info(" - input path: " + args.input);
    LOG.info(" - output path: " + args.output);
    LOG.info(" - number of reducers: " + args.numReducers);

    Configuration conf = getConf();
    conf.set("delimiter",args.delimiter);
    conf.set("exclude",args.exclude);
    conf.setInt("yidx",args.yidx);
    conf.set("labels",args.labels);
    conf.setLong("seed",args.seed);
    conf.setInt("sample",args.sample);
    conf.setInt("features",args.features);
    conf.setInt("numtrees",args.numtrees);
    conf.setInt("maxdepth",args.maxdepth);
    conf.setInt("minnodes",args.minnodes);
    Job job = Job.getInstance(conf);
    job.setJobName(TrainRandomForest.class.getSimpleName());
    job.setJarByClass(TrainRandomForest.class);

    job.setNumReduceTasks(args.numReducers);

    FileInputFormat.setInputPaths(job, new Path(args.input));
    FileOutputFormat.setOutputPath(job, new Path(args.output));

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(Text.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(Text.class);
    job.setOutputFormatClass(TextOutputFormat.class);

    job.setMapperClass(MyMapper.class);
    job.setReducerClass(MyReducer.class);

    // Delete the output directory if it exists already.
    Path outputDir = new Path(args.output);
    FileSystem.get(conf).delete(outputDir, true);

    long startTime = System.currentTimeMillis();
    job.waitForCompletion(true);
    LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
    return 0;
  }


  public static void main(String[] args) throws Exception {
    ToolRunner.run(new TrainRandomForest(), args);
  }
}
