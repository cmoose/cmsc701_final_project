package edu.umd;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import org.apache.commons.cli.*;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Created by chris on 11/3/15.
 */
public class CoreNLPprocessor {
    public static void main(String[] args) throws IOException {

        // create the Options
        Options options = new Options();
        options.addOption("v", false, "Verbose");
        options.addOption("inputfile", true, "Location of the input text file");
        options.addOption("outputfile", true, "Location of the output text file");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = null;
        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            e.printStackTrace();
        }

        // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);


        String textInputFilename = cmd.getOptionValue("inputfile");
        String textOutputFilename = cmd.getOptionValue("outputfile");

        ArrayList<String> all_text = new ArrayList<>();
        System.out.println("Loading inputfile into memory...");

        File textPath = new File(textInputFilename);
        if (textPath.isFile()) {
            BufferedReader reader = getBufferedReader(textPath);
            String line;
            while ((line = reader.readLine()) != null) {
                all_text.add(line);
            }
            reader.close();
        }

        BufferedWriter writer = getBufferedWriter(textOutputFilename);

        System.out.println("Lemmatizing all text...");
        int i=0;
        for (String text : all_text) {
            if (i % 1000 == 0) {
                System.out.printf(".");
            }

            Annotation document = new Annotation(text);

            // run all Annotators on this text
            pipeline.annotate(document);

            List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

            ArrayList<String> tokenArray = new ArrayList<>();
            for(CoreMap sentence: sentences) {
                // traversing the words in the current sentence
                // a CoreLabel is a CoreMap with additional token-specific methods
                for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {

                    String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                    tokenArray.add(token.lemma());
                }

            }

            String outLine = String.join(" ", tokenArray);
            writer.write(outLine);
            writer.newLine();
            i++;
        }

    }

    public static BufferedReader getBufferedReader(File file)
            throws FileNotFoundException, UnsupportedEncodingException {
        BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
        return in;
    }

    public static BufferedWriter getBufferedWriter(String filepath)
            throws FileNotFoundException, UnsupportedEncodingException {
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filepath), "UTF-8"));
        return out;
    }


    static void addOption(String optName, String optDesc) {
        Options options = new Options();
        options.addOption(OptionBuilder.withLongOpt(optName)
                .withDescription(optDesc)
                .hasArg()
                .withArgName(optName)
                .create());
    }
}

