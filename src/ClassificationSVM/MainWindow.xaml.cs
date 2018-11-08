using Accord.Imaging;
using Accord.Imaging.Filters;
using Accord.MachineLearning;
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
using System.Threading;
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
                dialog.ShowNewFolderButton = false;
                dialog.RootFolder = Environment.SpecialFolder.UserProfile;
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
                var bow = CreateBow();
                foreach (var image in Images)
                    TrainingData.Add(GetData(image, bow));

                var kernel = new Polynomial();
                var teacher = new MulticlassSupportVectorLearning<IKernel>()
                {
                    Kernel = kernel,
                    Learner = (param) => new SequentialMinimalOptimization<IKernel>()
                    {
                        Kernel = kernel
                    }
                };

                var svm = teacher.Learn(TrainingData.ToArray(), Tags.ToArray());
                var error = new ZeroOneLoss(Tags.ToArray()).Loss(svm.Decide(TrainingData.ToArray()));
                Error.Dispatcher.Invoke(() => Error.Text = error.ToString());

                //var kernel = new Polynomial(16, 5);
                //var complexity = CalculateComplexity(kernel);
                //var ml = new MulticlassSupportVectorLearning<IKernel>()
                //{
                //    Learner = (param) => new SequentialMinimalOptimization<IKernel>()
                //    {
                //        Complexity = complexity,
                //        Kernel = kernel
                //    }
                //};

                //machine = ml.Learn(TrainingData.ToArray(), Tags.ToArray());
                //var result = machine.Decide(TrainingData.ToArray());
                //var error = new ZeroOneLoss(Tags.ToArray())
                //{
                //    Mean = true
                //}.Loss(result);

                //Error.Dispatcher.Invoke(() => Error.Text = error.ToString());
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
        List<Bitmap> Images = new List<Bitmap>();

        private double CalculateComplexity(IKernel kernel)
            => kernel.EstimateComplexity(TrainingData.ToArray());

        private void LoadFiles(string mainDirectoryPath)
        {
            foreach (var directory in Directory.GetDirectories(mainDirectoryPath))
                LoadImagesV2(directory);
        }

        private void LoadImages(string directoryPath)
        {
            foreach (var file in Directory.GetFiles(directoryPath))
            {
                Tags.Add(GetClass(file));
                TrainingData.Add(LoadImage(file));
            }
        }

        private void LoadImagesV2(string directoryPath)
        {
            foreach (var file in Directory.GetFiles(directoryPath))
            {
                Tags.Add(GetClass(file));
                LoadImageV2(file);
            }
        }

        private int GetClass(string imagePath)
        {
            var containgFolder = System.IO.Path.GetDirectoryName(imagePath);
            var numStr = containgFolder.Split('\\').Last().Split('_')[0];
            return int.Parse(numStr.ToString()) - 1;
        }

        private IBagOfWords<Bitmap> CreateBow()
        {
            var binarySplit = new BinarySplit(10);
            var surfBow = BagOfVisualWords.Create(10);
            return surfBow.Learn(Images.ToArray());
        }

        private double[] GetData(Bitmap bitmap, IBagOfWords<Bitmap> bow)
            => (bow as ITransform<Bitmap, double[]>).Transform(bitmap);

        private void LoadImageV2(string imageFile)
        {
            using (var bitmap = new Bitmap(imageFile))
            {
                var processed = Scale(bitmap);
                Images.Add(bitmap.Clone() as Bitmap);
            }
        }

        private double[] LoadImage(string imageFile)
        {
            using (var bitmap = new Bitmap(imageFile))
            using (var ms = new MemoryStream())
            {
                var processed = Scale(bitmap);
                Images.Add(processed);
                ProccesdImage.Dispatcher.Invoke(() => ProccesdImage.Source = GetImage(Scale(processed)));

                Thread.Sleep(TimeSpan.FromSeconds(0.5));
                processed.Save(ms, ImageFormat.Bmp);
                var array = ms.ToArray();
                return array.Select(b => Convert.ToDouble(b)).ToArray();
            }
        }

        private BitmapImage GetImage(Bitmap bitmap)
        {
            using (var ms = new MemoryStream())
            {
                bitmap.Save(ms, ImageFormat.Bmp);
                ms.Position = 0;
                var bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = ms;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
                return bitmapImage;
            }
        }

        int width = 6;
        int height = 12;
        // w: 18, h: 44
        private Bitmap Scale(Bitmap input)
        {
            var destRect = new System.Drawing.Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(input.HorizontalResolution, input.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(input, destRect, 0, 0, input.Width, input.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }
    }
}
