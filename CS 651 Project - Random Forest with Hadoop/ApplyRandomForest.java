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
import org.apache.hadoop.mapreduce.Partitioner;
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
import java.util.regex.Pattern;
import java.net.URI;



public class ApplyRandomForest extends Configured implements Tool {
  private static final Logger LOG = Logger.getLogger(ApplyRandomForest.class);

  private static final class MyMapper extends Mapper<LongWritable, Text, IntWritable, DoubleWritable> {
    //start parsing data
    private static Text delimiter;
    private static IntArrayWritable exclude;
    private static IntWritable yidx;
    private static IntWritable ididx;
    private static Text[] labels;
    private static final ArrayList<HashMap<Long,Integer>> featuretrees = new ArrayList<>();
    private static final ArrayList<HashMap<Long,Double>> splittrees = new ArrayList<>();
    private static DoubleWritable crossentropy = new DoubleWritable(0.0);
    private static IntWritable totaltests = new IntWritable(0);
    private static IntWritable correcttests = new IntWritable(0);

    @Override
    public void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      delimiter = new Text(conf.get("delimiter"));
      String e0 = conf.get("exclude");
      String[] e1;
      if (e0==null) e1 = new String[0];
      else e1=e0.split(",");
      int[] e2 = new int[e1.length+2];
      for (int i=0; i<e1.length; i++) e2[i]=Integer.parseInt(e1[i]);
      yidx = new IntWritable(conf.getInt("yidx",1));
      ididx = new IntWritable(conf.getInt("ididx",0));
      e2[e1.length]=conf.getInt("yidx",1);
      e2[e1.length+1]=conf.getInt("ididx",0);
      Arrays.sort(e2);
      exclude = new IntArrayWritable(e2);
      String[] l = conf.get("labels").split(",");
      labels = new Text[l.length];
      for (int i=0; i<l.length; i++) labels[i] = new Text(l[i]);

      FileSystem fs = FileSystem.get(conf);
      for (int i=0; i<conf.getInt("lastReducers",0); i++) {
        LineReader reader = new LineReader(
          fs.open(new Path(context.getCacheFiles()[i].toString())));
        Text txt = new Text();
        Scanner in;
        while (reader.readLine(txt) > 0) {
          in=new Scanner(txt.toString());
          in.useDelimiter(Pattern.compile("[ {=,}]"));
          HashMap<Long,Integer> ft = new HashMap<>();
          HashMap<Long,Double> st = new HashMap<>();
          while(in.hasNext()) {
              in.next();
              long key = in.nextLong();
              ft.put(key,in.nextInt());
              st.put(key,in.nextDouble());
          }
          featuretrees.add(ft);
          splittrees.add(st);
        }
      }
    }
    @Override
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

      String[] tokens = value.toString().split(delimiter.toString());
      double[] x = new double[tokens.length-exclude.size()];
      int y = 0;
      int idx = 0;
      for (int i=0; i<tokens.length; i++) {
        if (Arrays.binarySearch(exclude.getArray(),i)<0) {
          x[idx]=Double.parseDouble(tokens[i]);
          idx++;
        }
      }
      for (int i=0; i<labels.length; i++) {
        if (tokens[yidx.get()].equals(labels[i].toString())) {
          y=i;
          break;
        }
      }
      int[] counts = new int[labels.length];
      for (int i=0; i<featuretrees.size(); i++) {
        HashMap<Long,Integer> ft = featuretrees.get(i);
        HashMap<Long,Double> st = splittrees.get(i);

        long k = 1;
        while (true) {
          int feature = ft.get(k);
          if (feature == -1) {
            counts[(int)(double)st.get(k)]++;
            break;
          }
          if (x[feature]<st.get(k)) k=2*k;
          else k=2*k+1;
        }
      }
      crossentropy.set(crossentropy.get()-Math.log((double)counts[y]/featuretrees.size()));
      totaltests.set(totaltests.get()+1);
      int pred = 0;
      for (int i=1; i<labels.length; i++) {
        if (counts[i]>counts[pred]) pred=i;
      }
      if (pred==y) correcttests.set(correcttests.get()+1);
      context.write(new IntWritable(Integer.parseInt(tokens[ididx.get()])), new DoubleWritable((double)pred));
    }
    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
      context.write(new IntWritable(-1),crossentropy);
      context.write(new IntWritable(-2),new DoubleWritable((double)totaltests.get()));
      context.write(new IntWritable(-3),new DoubleWritable((double)correcttests.get()));
    }
  }

  public static final class MyReducer extends Reducer<IntWritable, DoubleWritable, IntWritable, Text> {
    private static DoubleWritable crossentropy = new DoubleWritable(0.0);
    private static IntWritable totaltests = new IntWritable(0);
    private static IntWritable correcttests = new IntWritable(0);
    private static Text[] labels;

    @Override
    public void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      String[] l = conf.get("labels").split(",");
      labels = new Text[l.length];
      for (int i=0; i<l.length; i++) labels[i] = new Text(l[i]);
    }
    @Override
    public void reduce(IntWritable key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
      if (key.get()<0) {
        if (key.get()==-1) for (DoubleWritable i : values) crossentropy.set(crossentropy.get()+i.get());
        else if (key.get()==-2) for (DoubleWritable i : values) totaltests.set(totaltests.get()+(int)i.get());
        else for (DoubleWritable i : values) correcttests.set(correcttests.get()+(int)i.get());
        return;
      }
      for (DoubleWritable i : values) {
        context.write(key, labels[(int)i.get()]);
      }
    }
    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
      if (totaltests.get()!=0) {
        LOG.info("cross-entropy: "+crossentropy.get());
        LOG.info("accuracy: "+(double)correcttests.get()/totaltests.get());
      }
    }

  }
  private static final class MyPartitioner extends Partitioner<IntWritable, DoubleWritable> {
        @Override
        public int getPartition(IntWritable key, DoubleWritable value, int numPartitions) {
          if (key.get()<0) return numPartitions-1;
          int part = key.get() % (numPartitions-1);
          if (part<0) return part+(numPartitions-1);
          else return part;
        }
      }



  private ApplyRandomForest() {}

  private static final class Args {
    // io
    @Option(name = "-input", metaVar = "[path]", required = true, usage = "input path")
    String input;

    @Option(name = "-sidedata", metaVar = "[path]", required = true, usage = "sidedata path")
    String sidedata;

    @Option(name = "-output", metaVar = "[path]", required = true, usage = "output path")
    String output;

    @Option(name = "-reducers", metaVar = "[num]", usage = "number of reducers")
    int numReducers = 1;

    @Option(name = "-lastreducers", metaVar = "[num]", usage = "number of reducers used for training")
    int lastReducers = 0; //must include if different from numReducers

    // parse
    @Option(name = "-delimiter", metaVar = "[string]", usage = "delimiter to split lines")
    String delimiter = ",";

    @Option(name = "-exclude", metaVar = "[string]", usage = "columns to exclude for features")
    String exclude = "";

    @Option(name = "-yidx", metaVar = "[num]", usage = "index of response")
    int yidx = 1;

    @Option(name = "-ididx", metaVar = "[num]", usage = "index of each observation's ID")
    int ididx = 0;

    @Option(name = "-labels", metaVar = "[string]", usage = "name of labels")
    String labels = "0,1";
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

    LOG.info("Tool: " + ApplyRandomForest.class.getSimpleName());
    LOG.info(" - input path: " + args.input);
    LOG.info(" - sidedata path: " + args.sidedata);
    LOG.info(" - output path: " + args.output);
    LOG.info(" - number of reducers: " + args.numReducers);

    Configuration conf = getConf();
    conf.set("delimiter",args.delimiter);
    conf.set("exclude",args.exclude);
    conf.setInt("yidx",args.yidx);
    conf.setInt("ididx",args.ididx);
    conf.set("labels",args.labels);
    int lastReducers=0;
    if (args.lastReducers==0) lastReducers=args.numReducers;
    else lastReducers=args.lastReducers;
    conf.setInt("lastReducers",lastReducers);

    Job job = Job.getInstance(conf);
    job.setJobName(ApplyRandomForest.class.getSimpleName());
    job.setJarByClass(ApplyRandomForest.class);

    job.setNumReduceTasks(args.numReducers+1);

    FileInputFormat.setInputPaths(job, new Path(args.input));
    FileOutputFormat.setOutputPath(job, new Path(args.output));


    String outfile;
    for (int i=0; i<lastReducers; i++) {
      outfile=args.sidedata+"/part-r-"+(i/10000)+(i%10000/1000)+(i%1000/100)+(i%100/10)+(i%10);
      job.addCacheFile(new URI(outfile));
    }

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(DoubleWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
    job.setOutputFormatClass(TextOutputFormat.class);

    job.setMapperClass(MyMapper.class);
    job.setReducerClass(MyReducer.class);
    job.setPartitionerClass(MyPartitioner.class);


    // Delete the output directory if it exists already.
    Path outputDir = new Path(args.output);
    FileSystem.get(conf).delete(outputDir, true);

    long startTime = System.currentTimeMillis();
    job.waitForCompletion(true);
    LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
    return 0;
  }


  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ApplyRandomForest(), args);
  }
}
