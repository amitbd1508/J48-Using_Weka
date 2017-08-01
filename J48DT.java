
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class J48DT {

    public static void main(String[] args) throws Exception {

        String dir="F:\\weka\\cartest\\";
        Instances test = new Instances(new BufferedReader(new FileReader(dir+"test.arff")));
        Instances train = new Instances(new BufferedReader(new FileReader(dir+"train.arff")));
        FileWriter fileWriter = new FileWriter(dir+"newdataset.arff");
        String newLine = System.getProperty("line.separator");


        Classifier j48tree = new J48();
        int lastIndex = train.numAttributes() - 1;
        train.setClassIndex(lastIndex);
        test.setClassIndex(lastIndex);
        j48tree.buildClassifier(train);
        for(int i=0; i<test.numInstances(); i++) {
            double index = j48tree.classifyInstance(test.instance(i));
            String className = train.attribute(lastIndex).value((int)index);
            String newInstance="";
            for(int f=0;f< lastIndex;f++)
            {
                newInstance+=test.instance(i).stringValue(f)+",";
            }
            newInstance+=className;
            fileWriter.write(newInstance+newLine);


        }
        fileWriter.close();
    }
}
