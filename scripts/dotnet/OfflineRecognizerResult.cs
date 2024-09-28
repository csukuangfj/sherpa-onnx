/// Copyright (c)  2024.5 by 东风破

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{

    public class OfflineRecognizerResult
    {
        public OfflineRecognizerResult(IntPtr handle)
        {
            Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));

            // PtrToStringUTF8() requires .net standard 2.1
            // _text = Marshal.PtrToStringUTF8(impl.Text);

            int length = 0;

            unsafe
            {
                byte* buffer = (byte*)impl.Text;
                while (*buffer != 0)
                {
                    ++buffer;
                    length += 1;
                }
            }

            byte[] stringBuffer = new byte[length];
            Marshal.Copy(impl.Text, stringBuffer, 0, length);
            _text = Encoding.UTF8.GetString(stringBuffer);

            _tokens = new String[impl.Count];

            unsafe
            {
                byte* buf = (byte*)impl.Tokens;
                for (int i = 0; i < impl.Count; i++)
                {
                    length = 0;
                    byte* start = buf;
                    while (*buf != 0)
                    {
                        ++buf;
                        length += 1;
                    }
                    ++buf;

                    stringBuffer = new byte[length];
                    fixed (byte* pTarget = stringBuffer)
                    {
                        for (int k = 0; k < length; k++)
                        {
                            pTarget[k] = start[k];
                        }
                    }

                    _tokens[i] = Encoding.UTF8.GetString(stringBuffer);
                }
            }

            unsafe
            {
                float* t = (float*)impl.Timestamps;
                if (t != null)
                {
                    _timestamps = new float[impl.Count];
                    fixed (float* pTarget = _timestamps)
                    {
                        for (int i = 0; i < impl.Count; i++)
                        {
                            pTarget[i] = t[i];
                            Console.WriteLine(pTarget[i]);
                        }
                    }
                  Console.WriteLine("is not empty");
                  Console.WriteLine(impl.Count);
                  Console.WriteLine(_timestamps.Length);
                  Console.WriteLine(Timestamps.Length);
                  Console.WriteLine(_timestamps);
                  Console.WriteLine(Timestamps);
                }
                else
                {
                    Console.WriteLine("is empty");
                    _timestamps = new float[] {};
                }

                Console.WriteLine("count");
                Console.WriteLine(impl.Count);
            }

            _lang = "";
            _emotion = "";
            _event = "";

            unsafe
            {
                length = 0;
                byte* buffer = (byte*)impl.Lang;
                if (buffer != null)
                {
                  while (*buffer != 0)
                  {
                      ++buffer;
                      length += 1;
                  }
                  stringBuffer = new byte[length];
                  Marshal.Copy(impl.Lang, stringBuffer, 0, length);
                  _lang = Encoding.UTF8.GetString(stringBuffer);
                }
            }

            unsafe
            {
                length = 0;
                byte* buffer = (byte*)impl.Emotion;
                if (buffer != null)
                {
                  while (*buffer != 0)
                  {
                      ++buffer;
                      length += 1;
                  }
                  stringBuffer = new byte[length];
                  Marshal.Copy(impl.Emotion, stringBuffer, 0, length);
                  _emotion = Encoding.UTF8.GetString(stringBuffer);
                }
            }

            unsafe
            {
                length = 0;
                byte* buffer = (byte*)impl.Event;
                if (buffer != null)
                {
                  while (*buffer != 0)
                  {
                      ++buffer;
                      length += 1;
                  }
                  stringBuffer = new byte[length];
                  Marshal.Copy(impl.Event, stringBuffer, 0, length);
                  _event = Encoding.UTF8.GetString(stringBuffer);
                }
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public IntPtr Text;
            public IntPtr Timestamps;
            public int Count;
            public IntPtr Tokens;
            public IntPtr TokensArr;
            public IntPtr Json;
            public IntPtr Lang;
            public IntPtr Emotion;
            public IntPtr Event;
        }

        private String _text;
        public String Text => _text;

        private String[] _tokens;
        public String[] Tokens => _tokens;

        private float[] _timestamps;
        public float[] Timestamps => _timestamps;

        private String _lang;
        public String Lang => _lang;

        private String _emotion;
        public String Emotion => _emotion;

        private String _event;
        public String Event => _event;
    }
}
