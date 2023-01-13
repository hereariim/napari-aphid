t1 = getTime();
outputDataset = getArgument();
compressionLevel = 0;

dir = getArgument();
subfolders = getFileList(dir);
print("\\Clear");
setBatchMode(true);
for (i = 0; i < subfolders.length; i++) {
	showProgress(i+1, subfolders.length);
	print("Entering folder " + subfolders[i]);
	folder = dir + subfolders[i];
	files = getFileList(folder);
	for (f = 0; f < files.length; f++) {
		file = files[f];
		filesString = String.join(files, ",");
		if (endsWith(file, ".h5")) continue;
		out = replace(file, "png", "h5");
		if (indexOf(filesString, out)>-1) continue;
		print("Processing file " + file);
		open(folder + file);
		outputPath = folder + File.nameWithoutExtension + ".h5";			
		exportArgs = "select=" + outputPath + " datasetname=" + outputDataset + " compressionlevel=" + compressionLevel;
		run("Export HDF5", exportArgs);
		close("*");
		if (f%10==0) run("Collect Garbage");
	}
}
setBatchMode(false);
t2 = getTime();
print("batch convert to h5 took: " + (t2-t1)/1000 + "s");
