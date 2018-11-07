using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Kernels;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Forms;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ClassificationSVM
{
    public enum Classes
    {
        Black,
        White,
        Full,
        Quaver,
        SemiQuaver,
        QuaverL,
        QuaverM,
        QuaverR,
        SemiQuaverL,
        SemiQuaverM,
        SemiQuaverR,
        SemiQuaverTimeChange,
        TrebleClef,
        BassClef
    }

    /// <summary>
    /// Lógica de interacción para MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private MulticlassSupportVectorMachine<IKernel> machine = null;

        public void OpenDirectory(object sender, EventArgs args)
        {
            using (var dialog = new FolderBrowserDialog())
            {
                var result = dialog.ShowDialog();
                if(result == System.Windows.Forms.DialogResult.OK)
                {
                    Task.Factory.StartNew(() => LoadFiles(dialog.SelectedPath));
                }
            }
        }
        
        public void Learn_Clicked(object sender, EventArgs args)
        {
            Task.Factory.StartNew(() =>
            {
                var kernel = new Polynomial(16, 5);
                var complexity = CalculateComplexity(kernel);
                var ml = new MulticlassSupportVectorLearning<IKernel>()
                {
                    Learner = (param) => new SequentialMinimalOptimization<IKernel>()
                    {
                        Complexity = complexity,
                        Kernel = kernel
                    }
                };

                machine = ml.Learn(TrainingData.ToArray(), Tags.ToArray());
                var result = machine.Decide(TrainingData.ToArray());
                var error = new ZeroOneLoss(Tags.ToArray())
                {
                    Mean = true
                }.Loss(result);

                Error.Dispatcher.Invoke(() => Error.Text = error.ToString());
            });
        }

        public void Decide(object sender, EventArgs args)
        {
            using (var dlg = new OpenFileDialog())
            {
                var result = dlg.ShowDialog();
                if (result == System.Windows.Forms.DialogResult.OK)
                {
                    var loadedImage = LoadImage(dlg.FileName);
                    var machineResult = machine.Decide(loadedImage);
                    Result.Dispatcher.Invoke(() => Result.Text = machineResult.ToString());
                }
            }
        }

        List<int> Tags = new List<int>();
        List<double[]> TrainingData = new List<double[]>();

        private double CalculateComplexity(IKernel kernel)
            => kernel.EstimateComplexity(TrainingData.ToArray());

        private void LoadFiles(string mainDirectoryPath)
        {
            foreach (var directory in Directory.GetDirectories(mainDirectoryPath))
                LoadImages(directory);
        }

        private void LoadImages(string directoryPath)
        {
            foreach (var file in Directory.GetFiles(directoryPath))
            {
                Tags.Add(GetClass(file));
                TrainingData.Add(LoadImage(file));
            }
        }

        private int GetClass(string imagePath)
        {
            var containgFolder = System.IO.Path.GetDirectoryName(imagePath);
            var numStr = containgFolder.Split('\\').Last().Split('_')[0];
            return int.Parse(numStr.ToString()) - 1;
        }

        private double[] LoadImage(string imageFile)
        {
            using (var bitmap = new Bitmap(imageFile))
            using (var ms = new MemoryStream())
            {
                var newBitamp = ScaleImage(bitmap);
                newBitamp.Save(ms, ImageFormat.Bmp);
                var array = ms.ToArray();
                return array.Select(b => Convert.ToDouble(b)).ToArray();
            }
        }

        float width = 30;
        float height = 30;
        SolidBrush brush = new SolidBrush(System.Drawing.Color.Black);
        private Bitmap ScaleImage(Bitmap input)
        {
            var scale = Math.Min(width / input.Width, height / input.Height);
            var bmp = new Bitmap((int)width, (int)height);
            var graph = Graphics.FromImage(bmp);
            graph.InterpolationMode = InterpolationMode.High;
            graph.CompositingQuality = CompositingQuality.HighQuality;
            graph.SmoothingMode = SmoothingMode.AntiAlias;

            var scaleWidth = (int)(input.Width * scale);
            var scaleHeight = (int)(input.Height * scale);
            graph.FillRectangle(brush, new RectangleF(0, 0, width, height));
            graph.DrawImage(input, ((int)width - scaleWidth) / 2, ((int)height - scaleHeight) / 2, scaleWidth, scaleHeight);
            return bmp;
        }
    }
}
